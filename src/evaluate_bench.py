"""
Score the trackers and the speed estimates on the test bench, where the truth is known.

Tracking metrics (standard CLEAR-MOT / ID metrics, py-motmetrics), inside the evaluation zone:
  MOTA        overall tracking accuracy: 1 - (missed + false + ID switches) / true objects
  IDF1        how consistently each vehicle keeps one ID (most relevant for counting and speed)
  ID switches how often a vehicle's ID jumps to another one

Speed: each tracked vehicle is matched to the true vehicle it overlaps most, and its measured speed
is compared with the true speed (km/h).

    python src/evaluate_bench.py
"""

import json
from pathlib import Path

import matplotlib
import motmetrics as mm
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from road import ZONE, Road, add_road_coords, vehicle_speeds
from tracking import detect_video, run_tracker

ROOT = Path(__file__).resolve().parent.parent
OUT, ASSETS = ROOT / "results" / "bench", ROOT / "assets"
FPS = 30
EVAL_Y = (-10.0, 50.0)   # evaluation zone along the road (m); far beyond it vehicles are a few pixels wide
TRACKERS = {  # (tracker, measure on raw detections?) -> label
    ("centroid", True): "Centroid (classic OpenCV)",
    ("bytetrack", False): "ByteTrack, smoothed boxes",
    ("bytetrack", True): "ByteTrack",
    ("botsort", False): "BoT-SORT, smoothed boxes",
    ("botsort", True): "BoT-SORT",
}


def in_zone(df, road):
    Y = road.to_road(np.c_[(df.x1 + df.x2) / 2, df.y2])[:, 1]
    return df[(Y >= EVAL_Y[0]) & (Y <= EVAL_Y[1])]


def to_xywh(df):
    return np.c_[df.x1, df.y1, df.x2 - df.x1, df.y2 - df.y1]


def iou_distance(a, b, max_d=0.5):
    """1 - IoU between xywh boxes; NaN = not allowed to match (IoU below 0.5). Same convention as
    motmetrics.distances.iou_matrix, which fails on NumPy 2."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], (a[:, 0] + a[:, 2])[:, None], (a[:, 1] + a[:, 3])[:, None]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    inter = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None) * np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    union = a[:, 2, None] * a[:, 3, None] + b[:, 2] * b[:, 3] - inter
    d = 1 - inter / union
    d[d > max_d] = np.nan
    return d


def mot_scores(gt, tr, n_frames):
    acc = mm.MOTAccumulator(auto_id=False)
    g_by, t_by = dict(tuple(gt.groupby("frame"))), dict(tuple(tr.groupby("frame")))
    for f in range(n_frames):
        g, t = g_by.get(f, gt.iloc[:0]), t_by.get(f, tr.iloc[:0])
        dist = iou_distance(to_xywh(g), to_xywh(t))
        acc.update(g.vehicle_id.tolist(), t.track_id.tolist(), dist, frameid=f)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "recall", "precision", "num_false_positives",
                                 "num_misses", "mostly_tracked", "num_unique_objects"], name="x")
    return s.iloc[0].to_dict(), acc


def match_tracks(gt, tr):
    """Map every predicted track to the true vehicle it overlaps most (IoU > 0.3), by frame count."""
    m = gt.merge(tr, on="frame", suffixes=("_g", "_t"))
    ix = np.clip(np.minimum(m.x2_g, m.x2_t) - np.maximum(m.x1_g, m.x1_t), 0, None)
    iy = np.clip(np.minimum(m.y2_g, m.y2_t) - np.maximum(m.y1_g, m.y1_t), 0, None)
    inter = ix * iy
    union = (m.x2_g - m.x1_g) * (m.y2_g - m.y1_g) + (m.x2_t - m.x1_t) * (m.y2_t - m.y1_t) - inter
    m = m[inter / union > 0.3]
    best = m.groupby(["track_id", "vehicle_id"]).size().reset_index(name="n").sort_values("n", ascending=False)
    return best.drop_duplicates("track_id").set_index("track_id").vehicle_id


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    road = Road()
    gt_all = pd.read_csv(OUT / "ground_truth.csv")
    n_frames = int(gt_all.frame.max()) + 1
    dets = detect_video(ROOT / "data" / "bench" / "bench.mp4", cache=OUT / "detections.csv")
    gt = in_zone(gt_all, road)

    # detection recall by distance (is the detector the bottleneck?)
    g = gt.copy()
    g["Y"] = road.to_road(np.c_[(g.x1 + g.x2) / 2, g.y2])[:, 1]
    dd = dets[dets.conf >= 0.25]
    hits = []
    for f, gg in g.groupby("frame"):
        d = dd[dd.frame == f]
        for _, r in gg.iterrows():
            ix = np.clip(np.minimum(d.x2, r.x2) - np.maximum(d.x1, r.x1), 0, None)
            iy = np.clip(np.minimum(d.y2, r.y2) - np.maximum(d.y1, r.y1), 0, None)
            iou = ix * iy / ((r.x2 - r.x1) * (r.y2 - r.y1) + (d.x2 - d.x1) * (d.y2 - d.y1) - ix * iy)
            hits.append({"Y": r.Y, "hit": bool((iou > 0.5).any())})
    hits = pd.DataFrame(hits)
    bands = pd.cut(hits.Y, [-10, 0, 10, 20, 30, 40, 50])
    recall = hits.groupby(bands, observed=True).hit.mean().round(3)
    recall.index = [f"{int(i.left)} to {int(i.right)} m" for i in recall.index]
    recall.to_csv(OUT / "detection_recall_by_distance.csv", header=["recall"])

    rows, speed_rows = [], []
    true_speed = gt_all.groupby("vehicle_id").speed_kmh.first()
    true_lane = gt_all[(gt_all.Y >= ZONE[0]) & (gt_all.Y <= ZONE[1])].groupby("vehicle_id").lane.agg(lambda s: s.mode().iloc[0])
    for (kind, raw), label in TRACKERS.items():
        tr = run_tracker(kind, dets, n_frames, fps=FPS, raw_boxes=raw)
        tr.to_csv(OUT / f"tracks_{kind}{'' if raw else '_smoothed'}.csv", index=False)
        s, _ = mot_scores(gt, in_zone(tr, road), n_frames)
        tracks = add_road_coords(tr, road)
        sp = vehicle_speeds(tracks, FPS)
        counted = len(sp)                                  # every track with a speed counts as one vehicle
        mapping = match_tracks(gt, tr)
        sp["vehicle_id"] = sp.track_id.map(mapping)
        sp = sp.dropna(subset=["vehicle_id"])
        sp["true_kmh"] = sp.vehicle_id.map(true_speed)
        sp["true_lane"] = sp.vehicle_id.map(true_lane)
        sp["error_kmh"] = sp.speed_kmh - sp.true_kmh
        sp["tracker"] = label
        speed_rows.append(sp)
        measured = sp.vehicle_id.nunique()
        in_zone_truth = gt_all[(gt_all.Y >= ZONE[0]) & (gt_all.Y <= ZONE[1])].vehicle_id.nunique()
        rows.append({
            "tracker": label, "MOTA": round(s["mota"], 3), "IDF1": round(s["idf1"], 3),
            "ID switches": int(s["num_switches"]), "recall": round(s["recall"], 3),
            "precision": round(s["precision"], 3), "false positives": int(s["num_false_positives"]),
            "mostly tracked": f"{int(s['mostly_tracked'])}/{int(s['num_unique_objects'])}",
            "vehicles counted": f"{counted} (true {in_zone_truth})",
            "vehicles with a speed": f"{measured}/{in_zone_truth}",
            "speed MAE (km/h)": round(float(sp.error_kmh.abs().mean()), 2),
            "speed median abs error (km/h)": round(float(sp.error_kmh.abs().median()), 2),
            "speed bias (km/h)": round(float(sp.error_kmh.mean()), 2),
            "within ±5 km/h": f"{(sp.error_kmh.abs() <= 5).mean():.0%}",
            "lane correct": f"{(sp.lane == sp.true_lane).mean():.0%}",
        })
        print(rows[-1])
    # lane-change detection vs. truth (true change = vehicle whose true lane differs between start and end)
    # truth judged over the same reliable range the detector uses (from road.VALID_Y)
    from road import VALID_Y
    vis = gt_all[(gt_all.Y >= VALID_Y[0]) & (gt_all.Y <= VALID_Y[1])].sort_values("frame")
    first_last = vis.groupby("vehicle_id").agg(first=("X", "first"), last=("X", "last"))
    true_changes = set(first_last.index[(first_last["last"] - first_last["first"]).abs() > 2.5])
    lc_rows = []
    for label, sp in zip([r["tracker"] for r in rows], speed_rows):
        found = set(sp.loc[sp.lane_change, "vehicle_id"].astype(int))
        lc_rows.append({"tracker": label, "true lane changes": len(true_changes), "detected": len(found),
                        "correct": len(found & true_changes), "false": len(found - true_changes)})
    pd.DataFrame(lc_rows).to_csv(OUT / "lane_change_detection.csv", index=False)
    print(pd.DataFrame(lc_rows).to_string(index=False))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tracker_comparison.csv", index=False)
    speeds = pd.concat(speed_rows)
    speeds.to_csv(OUT / "speed_errors.csv", index=False)

    # charts
    best = table.sort_values("IDF1", ascending=False).iloc[0].tracker
    b = speeds[speeds.tracker == best]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = (75, 130)
    a1.plot(lim, lim, color="#999", ls="--", lw=1)
    a1.fill_between(lim, [lim[0] - 5, lim[1] - 5], [lim[0] + 5, lim[1] + 5], color="#2e9e44", alpha=0.1, label="±5 km/h")
    a1.scatter(b.true_kmh, b.speed_kmh, s=28, color="#1c69d4", label="one vehicle")
    a1.set(xlim=lim, ylim=lim, xlabel="true speed (km/h)", ylabel="measured speed (km/h)",
           title=f"Measured vs. true speed ({best}, {len(b)} vehicles)")
    a1.legend(frameon=False, loc="upper left", fontsize=8)
    cols = {"Centroid (classic OpenCV)": "#9aa5b8", "ByteTrack": "#1c69d4", "BoT-SORT": "#f08c00",
            "BoT-SORT, smoothed boxes": "#d6336c"}
    for name, gg in speeds[speeds.tracker.isin(cols)].groupby("tracker"):
        a2.hist(gg.error_kmh, bins=np.arange(-15, 15.5, 1), alpha=0.55, color=cols[name], label=name)
    a2.axvline(0, color="#333", lw=0.8)
    a2.set(xlabel="speed error (km/h)", ylabel="vehicles", title="Speed error by tracker")
    a2.legend(frameon=False, fontsize=8)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "bench_speed_accuracy.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(recall.index, recall.values * 100, color="#1c69d4")
    for i, v in enumerate(recall.values):
        ax.text(i, v * 100 + 1.5, f"{v:.0%}", ha="center", fontsize=9)
    ax.set(ylim=(0, 110), ylabel="vehicles detected (%)", xlabel="distance along the road from the reference line",
           title="Detection recall by distance (test bench)")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "bench_recall_by_distance.png", dpi=150)

    summary = {"trackers": rows, "detection_recall_by_distance": recall.to_dict(),
               "truth": {"vehicles": int(gt_all.vehicle_id.nunique()), "frames": n_frames}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + table.to_string(index=False))
    print("\nDetection recall by distance:\n" + recall.to_string())


if __name__ == "__main__":
    main()
