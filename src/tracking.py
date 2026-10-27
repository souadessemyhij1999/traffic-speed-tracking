"""
Detection (run once, cached) and three interchangeable multi-object trackers.

  * Centroid tracker - the classic OpenCV-tutorial approach: match each detection to the nearest
                       previous object centre (Hungarian assignment), drop objects unseen for a while
  * ByteTrack        - Kalman motion model + IoU matching, and a second pass that rescues
                       low-confidence detections (e.g. partly hidden vehicles)
  * BoT-SORT         - ByteTrack-style association with an improved Kalman state and camera-motion
                       compensation (switched off here: the camera is fixed)

All three get exactly the same detections, so differences come from the tracking logic alone.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO class ids
CARRIAGEWAY = np.array([[575, 0], [760, 0], [1280, 600], [1280, 720], [0, 720], [0, 650]])  # our side of the road


def detect_video(video, model="yolo26s.pt", imgsz=960, conf=0.1, cache=None):
    """Run the detector on every frame once and cache the boxes (CSV). Low conf on purpose:
    ByteTrack uses low-confidence boxes in its second matching pass; other trackers filter."""
    if cache and Path(cache).exists():
        return pd.read_csv(cache)
    det = YOLO(model)
    cap = cv2.VideoCapture(str(video))
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = det.predict(frame, imgsz=imgsz, conf=conf, classes=list(VEHICLES), verbose=False)[0]
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if cv2.pointPolygonTest(CARRIAGEWAY.astype(np.float32), ((x1 + x2) / 2, y2), False) >= 0:
                rows.append({"frame": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": c, "cls": int(k)})
        i += 1
        if i % 100 == 0:
            print(f"  detected {i} frames")
    df = pd.DataFrame(rows)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.round(2).to_csv(cache, index=False)
    return df


class CentroidTracker:
    """Classic centroid tracking: nearest-centre matching with a maximum jump and a patience counter."""

    def __init__(self, max_distance=60, max_missing=15, min_conf=0.25):
        self.max_distance, self.max_missing, self.min_conf = max_distance, max_missing, min_conf
        self.objects, self.missing, self.next_id = {}, {}, 1

    def update(self, boxes):
        boxes = boxes[boxes[:, 4] >= self.min_conf]
        cents = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        ids = list(self.objects)
        assigned = {}
        if ids and len(cents):
            prev = np.array([self.objects[k] for k in ids])
            dist = np.linalg.norm(prev[:, None] - cents[None], axis=2)
            r, c = linear_sum_assignment(dist)
            for i, j in zip(r, c):
                if dist[i, j] <= self.max_distance:
                    assigned[j] = ids[i]
        for j in range(len(cents)):
            if j not in assigned:
                assigned[j] = self.next_id
                self.next_id += 1
            self.objects[assigned[j]] = cents[j]
            self.missing[assigned[j]] = 0
        for k in [k for k in self.objects if k not in assigned.values()]:
            self.missing[k] += 1
            if self.missing[k] > self.max_missing:
                del self.objects[k], self.missing[k]
        return [(assigned[j], *boxes[j, :4]) for j in range(len(boxes))]


class UltralyticsTracker:
    def __init__(self, kind, frame_rate=30, raw_boxes=True):
        self.raw_boxes = raw_boxes
        # track_buffer is in frames: keep lost tracks for 1 second
        base = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=frame_rate,
                    match_thresh=0.8, fuse_score=True)
        if kind == "bytetrack":
            self.t = BYTETracker(SimpleNamespace(tracker_type="bytetrack", **base))
        else:
            self.t = BOTSORT(SimpleNamespace(tracker_type="botsort", gmc_method="none", proximity_thresh=0.5,
                                             appearance_thresh=0.8, with_reid=False, model="auto", **base))

    def update(self, boxes, shape):
        data = np.c_[boxes[:, :5], boxes[:, 5]] if len(boxes) else np.zeros((0, 6))
        out = self.t.update(Boxes(data, shape), img=None)
        # The tracker decides WHO is who; the position comes from the raw detection it matched (last column
        # = detection index). The tracker's own Kalman-smoothed box assumes constant speed in pixels, but
        # cars driving away slow down in the image (perspective), so smoothed boxes overshoot.
        if self.raw_boxes:
            return [(int(r[4]), *boxes[int(r[-1]), :4]) for r in out]
        return [(int(r[4]), *r[:4]) for r in out]


def run_tracker(kind, dets, n_frames, shape=(720, 1280), fps=30, raw_boxes=True):
    """Return one row per tracked object per frame."""
    tracker = CentroidTracker() if kind == "centroid" else UltralyticsTracker(kind, fps, raw_boxes)
    by_frame = {f: g[["x1", "y1", "x2", "y2", "conf", "cls"]].to_numpy(float) for f, g in dets.groupby("frame")}
    rows = []
    for f in range(n_frames):
        boxes = by_frame.get(f, np.zeros((0, 6)))
        out = tracker.update(boxes) if kind == "centroid" else tracker.update(boxes, shape)
        rows += [{"frame": f, "track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2} for tid, x1, y1, x2, y2 in out]
    return pd.DataFrame(rows)
