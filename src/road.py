"""
Road geometry and speed estimation.

A vehicle's position on the road is the point under the bottom-centre of its box (where its rear
meets the road), mapped from pixels to metres with the calibrated homography.

Speed per vehicle = slope of a robust (Theil-Sen) straight-line fit of distance-along-road against
time, using only points inside the measurement zone, where the calibration was checked.
Instantaneous speed for display = the same fit over a sliding 0.5 s window.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parent.parent
LANE = 3.658
LANE_EDGES = [-LANE, 0, LANE, 2 * LANE, 3 * LANE, 4 * LANE]   # 5 lanes, lane 1 = leftmost (fast lane)
ZONE = (-8.0, 30.0)   # measurement zone along the road (m): from just below the camera to the validated range


class Road:
    def __init__(self, path=ROOT / "configs" / "road_homography.yaml"):
        self.H = np.array(yaml.safe_load(open(path))["homography"], np.float64)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _apply(M, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        p = np.c_[pts, np.ones(len(pts))] @ M.T
        return p[:, :2] / p[:, 2:3]

    def to_road(self, pts):
        return self._apply(self.H, pts)

    def to_image(self, pts):
        return self._apply(self.Hinv, pts)

    def px_per_m(self, X, Y):
        """Image pixels per metre across the road at road position (X, Y): the local scale."""
        a, b = self.to_image([[X - 0.5, Y], [X + 0.5, Y]])
        return float(np.hypot(*(b - a)))

    @staticmethod
    def lane(X):
        i = np.searchsorted(LANE_EDGES, X) if np.ndim(X) == 0 else np.searchsorted(LANE_EDGES, np.asarray(X))
        return np.where((np.asarray(X) < LANE_EDGES[0]) | (np.asarray(X) > LANE_EDGES[-1]), 0, i)


def add_road_coords(tracks: pd.DataFrame, road: Road, shape=(720, 1280)) -> pd.DataFrame:
    t = tracks.copy()
    rc = road.to_road(np.c_[(t.x1 + t.x2) / 2, t.y2])
    t["X"], t["Y"] = rc[:, 0], rc[:, 1]
    h, w = shape
    # a box cut off by the image border doesn't show the true contact point
    t["truncated"] = (t.y2 >= h - 3) | (t.x1 <= 2) | (t.x2 >= w - 3)
    t["lane"] = Road.lane(t.X.to_numpy())
    return t


def instantaneous_speed(t: pd.DataFrame, fps: float, window=15) -> pd.DataFrame:
    """km/h from a sliding linear fit of Y over `window` frames (0.5 s at 30 fps)."""
    t = t.sort_values(["track_id", "frame"]).copy()
    speeds = np.full(len(t), np.nan)
    for _, idx in t.groupby("track_id").indices.items():
        g = t.iloc[idx]
        if len(g) < window:
            continue
        sec, Y = g.frame.to_numpy() / fps, g.Y.to_numpy()
        half = window // 2
        for k in range(half, len(g) - half):
            s = slice(k - half, k + half + 1)
            speeds[idx[k]] = np.polyfit(sec[s], Y[s], 1)[0] * 3.6
    t["speed_kmh"] = speeds
    return t


LANE_CHANGE_MIN_SHIFT = 2.5   # m sideways: most of a 3.66 m lane, so drifting along a lane line doesn't count
VALID_Y = (-8.0, 45.0)        # range where lateral positions are reliable enough to judge lane changes


def lane_change(g: pd.DataFrame, fps: float):
    """(changed?, sideways shift in m) from the first and last 0.3 s of the track's reliable part."""
    v = g[(g.Y >= VALID_Y[0]) & (g.Y <= VALID_Y[1]) & ~g.truncated].sort_values("frame")
    n = max(int(0.3 * fps), 5)            # short windows: at 30 m/s one second already covers 30 m of road
    if len(v) < 3 * n:
        return False, 0.0
    x = v.X.rolling(n, center=True, min_periods=1).median()   # smooth out box jitter
    x0, x1 = x.iloc[:n].median(), x.iloc[-n:].median()
    shift = x1 - x0
    changed = abs(shift) >= LANE_CHANGE_MIN_SHIFT and Road.lane(x0) != Road.lane(x1)
    return bool(changed), round(float(shift), 2)


def vehicle_speeds(t: pd.DataFrame, fps: float, min_seconds=0.5) -> pd.DataFrame:
    """One robust speed per tracked vehicle, measured inside the calibrated zone."""
    rows = []
    for tid, g in t.groupby("track_id"):
        z = g[(g.Y >= ZONE[0]) & (g.Y <= ZONE[1]) & ~g.truncated]
        if len(z) < min_seconds * fps:
            continue
        slope, _, lo, hi = theilslopes(z.Y.to_numpy(), z.frame.to_numpy() / fps)
        lanes = z.lane[z.lane > 0]
        changed, shift = lane_change(g, fps)
        rows.append({
            "track_id": tid, "speed_kmh": round(slope * 3.6, 1),
            "speed_ci_kmh": round((hi - lo) * 3.6 / 2, 1),
            "lane": int(lanes.mode().iloc[0]) if len(lanes) else 0,
            "lanes_used": "-".join(str(int(x)) for x in pd.unique(lanes)) if len(lanes) else "",
            "first_frame": int(g.frame.min()), "last_frame": int(g.frame.max()),
            "seconds_in_zone": round(len(z) / fps, 2),
            "lane_change": changed, "sideways_shift_m": shift,
        })
    return pd.DataFrame(rows)
