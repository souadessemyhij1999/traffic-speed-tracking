"""
Semi-synthetic test bench with exact ground truth.

Real traffic video has no "true" speeds or IDs to score against. So, as in simulation-based testing
in automotive validation, this script builds a test video where everything is known:

  * background = the real empty road (median of the clip)
  * vehicles   = real vehicle cut-outs taken from the clip (vehicle pixels separated from the empty
                 road), so the detector sees real-looking cars
  * motion     = lanes, speeds (km/h), spawn times and lane changes drawn at random, then rendered with
                 the calibrated homography, so every car shrinks with distance exactly as the camera
                 would see it, and nearer cars cover farther ones

Writes data/bench/bench.mp4 and results/bench/ground_truth.csv (true box, road position and speed of
every vehicle in every frame).

    python src/bench.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from road import LANE, LANE_EDGES, Road

ROOT = Path(__file__).resolve().parent.parent
FPS, SECONDS, SEED = 30, 25, 7
W, H = 1280, 720
LANE_SPEED_KMH = [118, 110, 102, 95, 88]    # mean speed per lane, fast lane on the left
SPAWN_Y, EXIT_Y = -14.0, 85.0               # metres along the road


def cut_out_sprites(road, max_sprites=30):
    """Take well-visible vehicles from the real clip and separate them from the empty road."""
    bg = cv2.imread(str(ROOT / "data" / "background.png"))
    dets = pd.read_csv(ROOT / "results" / "freeway" / "detections.csv")
    d = dets[(dets.conf > 0.6) & (dets.y2 > 400) & (dets.y2 < 700) & (dets.x1 > 5) & (dets.x2 < W - 5)].copy()
    d["cx"] = ((d.x1 + d.x2) / 2 / 160).astype(int)      # spread sprites over the image: one per region...
    d = d.sort_values("conf", ascending=False).groupby(["cx", (d.frame // 90)]).head(1)   # ...and per 3 s
    cap = cv2.VideoCapture(str(ROOT / "data" / "freeway.mp4"))
    sprites = []
    for _, r in d.iterrows():
        # skip boxes that overlap another vehicle in the same frame
        same = dets[(dets.frame == r.frame) & (dets.conf > 0.3)]
        ox = np.minimum(same.x2, r.x2) - np.maximum(same.x1, r.x1)
        oy = np.minimum(same.y2, r.y2) - np.maximum(same.y1, r.y1)
        if ((ox > 0) & (oy > 0)).sum() > 1:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = int(r.x1) - 4, int(r.y1) - 4, int(r.x2) + 4, int(r.y2) + 4
        crop, under = frame[y1:y2, x1:x2], bg[y1:y2, x1:x2]
        diff = np.abs(crop.astype(int) - under.astype(int)).max(axis=2)
        mask = (diff > 28).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            continue
        mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)   # the vehicle blob
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cnts, -1, 1, -1)                                       # fill holes (windows)
        if mask.mean() < 0.35:
            continue
        ys, xs = np.where(mask)
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        bx = (xs.min() + xs.max()) / 2
        X, Y = road.to_road([[x1 + bx, y1 + ys.max()]])[0]
        sprites.append({"img": crop, "alpha": alpha, "anchor": (bx, ys.max()),   # bottom-centre = road contact
                        "ppm": road.px_per_m(X, Y), "cls": int(r.cls)})
        if len(sprites) >= max_sprites:
            break
    return sprites


def make_scenario(rng, sprites):
    vehicles, vid = [], 0
    for lane in range(5):
        t = rng.uniform(0, 2)
        last_speed = None
        while t < SECONDS - 1:
            speed = rng.normal(LANE_SPEED_KMH[lane], 4)
            if last_speed is not None:
                speed = min(speed, last_speed)          # no driving through the car ahead in the same lane
            change = None
            if rng.random() < 0.35:                     # 35% change lane once, inside the measurable range
                target = lane + rng.choice([-1, 1])
                if 0 <= target < 5:
                    change = (t + rng.uniform(0.1, 0.6), target, 1.5)   # start time, target lane, duration (s)
            vehicles.append({"id": vid, "lane": lane, "speed_kmh": speed, "t0": t, "change": change,
                             "sprite": int(rng.integers(len(sprites)))})
            vid += 1
            last_speed = speed
            t += max(rng.exponential(3.0), 1.4)         # time gap to the next vehicle in this lane
    return vehicles


def lane_centre(i):
    return (LANE_EDGES[i] + LANE_EDGES[i + 1]) / 2


def position(v, t):
    dt = t - v["t0"]
    Y = SPAWN_Y + v["speed_kmh"] / 3.6 * dt
    X = lane_centre(v["lane"])
    if v["change"]:
        start, target, dur = v["change"]
        f = np.clip((t - start) / dur, 0, 1)
        f = 0.5 - 0.5 * np.cos(np.pi * f)               # smooth S-shaped lane change
        X = X + (lane_centre(target) - X) * f
    return X, Y


def main():
    rng = np.random.default_rng(SEED)
    road = Road()
    sprites = cut_out_sprites(road)
    print(f"{len(sprites)} vehicle cut-outs taken from the real clip")
    vehicles = make_scenario(rng, sprites)
    bg = cv2.imread(str(ROOT / "data" / "background.png")).astype(np.float32)

    out_dir = ROOT / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / "bench.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = []
    for f in range(FPS * SECONDS):
        t = f / FPS
        frame = bg.copy()
        live = []
        for v in vehicles:
            if t < v["t0"]:
                continue
            X, Y = position(v, t)
            if Y > EXIT_Y:
                continue
            live.append((Y, X, v))
        for Y, X, v in sorted(live, key=lambda a: -a[0]):          # far first, near last (occlusion)
            s = sprites[v["sprite"]]
            scale = road.px_per_m(X, Y) / s["ppm"]
            if scale < 0.04:
                continue
            img = cv2.resize(s["img"], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
            a = cv2.resize(s["alpha"], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)[..., None]
            px, py = road.to_image([[X, Y]])[0]
            ox, oy = int(round(px - s["anchor"][0] * scale)), int(round(py - s["anchor"][1] * scale))
            x1, y1, x2, y2 = max(ox, 0), max(oy, 0), min(ox + img.shape[1], W), min(oy + img.shape[0], H)
            if x2 <= x1 or y2 <= y1:
                continue
            sub_img, sub_a = img[y1 - oy:y2 - oy, x1 - ox:x2 - ox], a[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - sub_a) + sub_img * sub_a
            ys, xs = np.where(sub_a[..., 0] > 0.5)
            if len(xs) < 4:
                continue
            gt.append({"frame": f, "vehicle_id": v["id"], "x1": x1 + xs.min(), "y1": y1 + ys.min(),
                       "x2": x1 + xs.max(), "y2": y1 + ys.max(), "X": round(X, 3), "Y": round(Y, 3),
                       "lane": int(Road.lane(X)), "speed_kmh": round(v["speed_kmh"], 2)})
        noisy = frame + rng.normal(0, 2.0, frame.shape)               # a little sensor noise
        writer.write(np.clip(noisy, 0, 255).astype(np.uint8))
    writer.release()
    gt = pd.DataFrame(gt)
    (ROOT / "results" / "bench").mkdir(parents=True, exist_ok=True)
    gt.to_csv(ROOT / "results" / "bench" / "ground_truth.csv", index=False)
    changes = sum(1 for v in vehicles if v["change"] and v["t0"] < SECONDS)
    print(f"Rendered {FPS * SECONDS} frames: {gt.vehicle_id.nunique()} vehicles, {changes} lane changes, "
          f"speeds {gt.speed_kmh.min():.0f}-{gt.speed_kmh.max():.0f} km/h")


if __name__ == "__main__":
    main()
