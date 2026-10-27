"""
Calibrate the camera-to-road mapping (a homography) from lane markings, so that every pixel on the
road gets real-world coordinates in metres.

US interstate markings follow a national standard (MUTCD): broken lane lines are 10 ft (3.05 m) long
with 30 ft (9.14 m) gaps, i.e. one dash every 12.19 m, and lanes are 12 ft (3.66 m) wide.

  1. Remove the traffic: the per-pixel median of frames across the clip leaves the empty road.
  2. Find the white dashes automatically (top-hat filter + connected components).
  3. Fit the homography on the 2 nearest rows of dashes (16 endpoints).
  4. Validate on the 3 farther rows the fit never saw: predicted dash length, dash spacing and lane width.

Writes configs/road_homography.yaml and assets/calibration.png

    python src/calibrate.py
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DASH, PERIOD, LANE = 3.048, 12.192, 3.658   # metres (10 ft, 40 ft, 12 ft)
CARRIAGEWAY = np.array([[575, 0], [745, 0], [1200, 720], [130, 720]])  # between the solid edge lines


def empty_road(video, step=12):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    return np.median(np.stack(frames), axis=0).astype(np.uint8), len(frames)


def find_dashes(bg):
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    white = (tophat > 35).astype(np.uint8)
    mask = np.zeros_like(white)
    cv2.fillPoly(mask, [CARRIAGEWAY], 1)
    white = cv2.morphologyEx(white & mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white)
    dashes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 25 or h < 8 or h > 150 or w > 60 or y + h > 700:  # ignore specks and the bridge rail at the bottom
            continue
        ys, xs = np.where(lab == i)
        top = (xs[ys == ys.min()].mean(), ys.min())
        bottom = (xs[ys == ys.max()].mean(), ys.max())
        length = np.hypot(top[0] - bottom[0], top[1] - bottom[1])
        if length / (a / max(length, 1)) < 3:  # a dash is long and thin; specks and blobs are not
            continue
        dashes.append({"top": top, "bottom": bottom, "length": length})
    # group into rows (same distance from the camera), nearest row first, each row sorted left to right
    dashes.sort(key=lambda d: -d["bottom"][1])
    rows = []
    for d in dashes:
        if rows and abs(rows[-1][0]["bottom"][1] - d["bottom"][1]) < 0.12 * d["bottom"][1] + 6:
            rows[-1].append(d)
        else:
            rows.append([d])
    clean = []
    for r in rows:
        if len(r) > 4:  # a stray mark joined the row: keep the 4 dashes whose length is most typical
            med = np.median([d["length"] for d in r])
            r = sorted(r, key=lambda d: abs(d["length"] - med))[:4]
        if len(r) == 4:  # rows where all 4 dashed lane lines are visible
            clean.append(sorted(r, key=lambda d: d["bottom"][0]))
    rows = clean
    return rows


def road_points(rows):
    """Image points and their road coordinates (X across the road, Y along it), in metres."""
    img, world = [], []
    for k, row in enumerate(rows):
        for line, d in enumerate(row):
            img += [d["bottom"], d["top"]]
            world += [(line * LANE, k * PERIOD), (line * LANE, k * PERIOD + DASH)]
    return np.array(img, np.float64), np.array(world, np.float64)


def main():
    bg, n_frames = empty_road(ROOT / "data" / "freeway.mp4")
    cv2.imwrite(str(ROOT / "data" / "background.png"), bg)
    rows = find_dashes(bg)
    print(f"Empty road from {n_frames} frames; {len(rows)} complete rows of dashes found")

    fit_rows, test_rows = rows[:2], rows[2:]
    img_fit, world_fit = road_points(fit_rows)
    H, _ = cv2.findHomography(img_fit, world_fit, 0)

    def to_road(pts):
        return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)

    fit_err = np.linalg.norm(to_road(img_fit) - world_fit, axis=1)
    # held-out validation: measure the far dashes with the calibration and compare with the standard
    checks = []
    for k, row in enumerate(test_rows, start=2):
        ends = [(to_road([d["bottom"]])[0], to_road([d["top"]])[0]) for d in row]
        for line, (b, t) in enumerate(ends):
            checks.append({"row": k, "line": line, "dash_length_m": float(t[1] - b[1]),
                           "dash_start_m": float(b[1]), "expected_start_m": k * PERIOD})
        xs = [b[0] for b, _ in ends]
        checks[-1]["lane_widths_m"] = [round(float(x), 2) for x in np.diff(xs)]
    lengths = np.array([c["dash_length_m"] for c in checks])
    pos_err = np.array([c["dash_start_m"] - c["expected_start_m"] for c in checks])
    widths = np.concatenate([c["lane_widths_m"] for c in checks if "lane_widths_m" in c])
    far = max(c["expected_start_m"] for c in checks) + DASH

    cfg = {
        "description": "Image (pixels) -> road plane (metres). X = across the road (0 = leftmost dashed line), "
                       "Y = along the road, away from the camera (0 = nearest calibrated dash).",
        "homography": [[round(float(v), 10) for v in r] for r in H],
        "calibration": {
            "standard": "MUTCD: 3.05 m dashes every 12.19 m, 3.66 m lanes",
            "fit_points": int(len(img_fit)), "fit_rms_error_m": round(float(np.sqrt((fit_err ** 2).mean())), 3),
            "held_out_rows": len(test_rows), "held_out_dashes": len(checks), "validated_up_to_m": round(far, 1),
            "held_out_dash_length_m": {"mean": round(float(lengths.mean()), 2), "expected": DASH,
                                       "max_abs_error": round(float(np.abs(lengths - DASH).max()), 2)},
            "held_out_dash_position_error_m": {"mean": round(float(pos_err.mean()), 2),
                                               "max_abs": round(float(np.abs(pos_err).max()), 2)},
            "held_out_lane_width_m": {"mean": round(float(widths.mean()), 2), "expected": LANE,
                                      "max_abs_error": round(float(np.abs(widths - LANE).max()), 2)},
        },
        "lanes": {"count": 5, "width_m": LANE, "x_left_edge_m": -LANE, "x_right_edge_m": 4 * LANE},
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    with open(ROOT / "configs" / "road_homography.yaml", "w") as f:
        f.write("# Produced by src/calibrate.py from the lane markings of the empty road\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(yaml.safe_dump(cfg["calibration"], sort_keys=False))

    # visual check: draw a 3.66 m x 10 m metric grid back onto the image with the inverse mapping
    Hinv = np.linalg.inv(H)
    vis = bg.copy()
    for x in np.arange(-LANE, 4 * LANE + 0.1, LANE):
        pts = cv2.perspectiveTransform(np.array([[[x, y] for y in np.linspace(-15, far + 30, 60)]], np.float64), Hinv)
        cv2.polylines(vis, [pts.astype(np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
    for y in range(-10, int(far) + 31, 10):
        pts = cv2.perspectiveTransform(np.array([[[x, y] for x in np.linspace(-LANE, 4 * LANE, 20)]], np.float64), Hinv)
        cv2.polylines(vis, [pts.astype(np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
        p = pts[0, -1].astype(int)
        cv2.putText(vis, f"{y} m", (p[0] + 6, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    for k, row in enumerate(rows):
        col = (60, 200, 60) if k < 2 else (60, 60, 230)
        for d in row:
            cv2.line(vis, tuple(map(int, d["bottom"])), tuple(map(int, d["top"])), col, 3)
    cv2.putText(vis, "green = dashes used for calibration   red = held-out dashes used to check it", (20, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    (ROOT / "assets").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "assets" / "calibration.png"), vis)


if __name__ == "__main__":
    main()
