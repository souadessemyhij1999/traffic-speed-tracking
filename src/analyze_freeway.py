"""
Track every vehicle in the real freeway clip, measure its speed in km/h and produce traffic analytics.

Outputs
  results/freeway/vehicles.csv        one row per vehicle: speed, lane, lane changes, time in view
  results/freeway/tracks.csv          one row per vehicle per frame: box, road position (m), speed
  results/freeway/lane_summary.csv    per lane: vehicles, flow, mean / median / 85th-percentile speed
  results/freeway/annotated.mp4       video with IDs, speeds, trails, counting line and a bird's-eye inset
  assets/...                          charts and demo animation for the README

    python src/analyze_freeway.py --tracker bytetrack
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from road import LANE_EDGES, ZONE, Road, add_road_coords, instantaneous_speed, vehicle_speeds
from tracking import detect_video, run_tracker

ROOT = Path(__file__).resolve().parent.parent
OUT, ASSETS = ROOT / "results" / "freeway", ROOT / "assets"
COUNT_LINE_Y = 10.0   # m: a vehicle is counted when it crosses this line


def speed_colour(kmh):
    """Green (slow) -> yellow -> red (fast), BGR."""
    if np.isnan(kmh):
        return (200, 200, 200)
    f = np.clip((kmh - 65) / 45, 0, 1)   # 65 km/h and below = green, 110 and above = red
    return (40, int(200 * (1 - f) + 60 * f), int(60 + 195 * f))


def draw_video(video, tracks, vehicles, road, fps, out_path):
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    by_frame = dict(tuple(tracks.groupby("frame")))
    counted, lane_counts = set(), {i: 0 for i in range(1, 6)}
    vspeed = vehicles.set_index("track_id").speed_kmh.to_dict()
    line = road.to_image([[LANE_EDGES[0], COUNT_LINE_Y], [LANE_EDGES[-1], COUNT_LINE_Y]]).astype(int)
    zone = np.vstack([road.to_image([[x, ZONE[0]] for x in (LANE_EDGES[0], LANE_EDGES[-1])]),
                      road.to_image([[x, ZONE[1]] for x in (LANE_EDGES[-1], LANE_EDGES[0])])]).astype(np.int32)
    snapshots, f = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (255, 200, 0))
        frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
        cv2.line(frame, tuple(line[0]), tuple(line[1]), (255, 255, 255), 2, cv2.LINE_AA)
        g = by_frame.get(f, tracks.iloc[:0])
        for _, r in g.iterrows():
            hist = tracks[(tracks.track_id == r.track_id) & (tracks.frame <= f) & (tracks.frame > f - 45)]
            pts = np.c_[(hist.x1 + hist.x2) / 2, hist.y2].astype(np.int32)
            # speed shown: live estimate inside the calibrated zone; afterwards the vehicle's measured zone speed
            if ZONE[0] <= r.Y <= ZONE[1]:
                shown = r.speed_kmh
            else:
                shown = vspeed.get(r.track_id, np.nan) if r.Y > ZONE[1] else np.nan
            col = speed_colour(shown)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, col, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
            label = f"#{r.track_id}" + ("" if np.isnan(shown) else f" {shown:.0f} km/h")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(r.x1), int(r.y1) - th - 8), (int(r.x1) + tw + 6, int(r.y1)), col, -1)
            cv2.putText(frame, label, (int(r.x1) + 3, int(r.y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            prev = tracks[(tracks.track_id == r.track_id) & (tracks.frame == f - 1)]
            if r.track_id not in counted and len(prev) and prev.Y.iloc[0] < COUNT_LINE_Y <= r.Y and 1 <= r.lane <= 5:
                counted.add(r.track_id)
                lane_counts[int(r.lane)] += 1
        # bird's-eye inset: 5 lanes x 0-60 m
        bw, bh, x0, y0 = 170, 300, W - 190, 20
        cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + bw + 8, y0 + bh + 34), (40, 40, 40), -1)
        sx, sy = bw / (LANE_EDGES[-1] - LANE_EDGES[0]), bh / 60
        for e in LANE_EDGES:
            x = int(x0 + (e - LANE_EDGES[0]) * sx)
            cv2.line(frame, (x, y0), (x, y0 + bh), (120, 120, 120), 1)
        for _, r in g.iterrows():
            if 0 <= r.Y <= 60 and LANE_EDGES[0] <= r.X <= LANE_EDGES[-1]:
                cx, cy = int(x0 + (r.X - LANE_EDGES[0]) * sx), int(y0 + bh - r.Y * sy)
                cv2.rectangle(frame, (cx - 5, cy - 9), (cx + 5, cy + 9), speed_colour(vspeed.get(r.track_id, r.speed_kmh)), -1)
        cv2.putText(frame, "bird's-eye 0-60 m", (x0, y0 + bh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        # counts panel
        cv2.rectangle(frame, (10, 10), (330, 64), (40, 40, 40), -1)
        cv2.putText(frame, f"Vehicles counted: {len(counted)}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "per lane: " + "  ".join(f"L{k} {v}" for k, v in lane_counts.items()), (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        writer.write(frame)
        if 300 <= f < 570 and f % 3 == 0:
            snapshots.append(frame)
        f += 1
    writer.release()
    return snapshots, lane_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracker", default="botsort", choices=["centroid", "bytetrack", "botsort"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    video = ROOT / "data" / "freeway.mp4"
    cap = cv2.VideoCapture(str(video))
    fps, n_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    road = Road()

    dets = detect_video(video, cache=OUT / "detections.csv")
    tr = run_tracker(args.tracker, dets, n_frames, fps=round(fps))
    tracks = instantaneous_speed(add_road_coords(tr, road), fps)
    vehicles = vehicle_speeds(tracks, fps)
    tracks.round(3).to_csv(OUT / "tracks.csv", index=False)
    vehicles.to_csv(OUT / "vehicles.csv", index=False)

    duration_min = n_frames / fps / 60
    lanes = vehicles[vehicles.lane > 0].groupby("lane").speed_kmh
    lane_summary = pd.DataFrame({
        "vehicles": lanes.size(), "flow_veh_per_hour": (lanes.size() / duration_min * 60).round(0),
        "mean_kmh": lanes.mean().round(1), "median_kmh": lanes.median().round(1),
        "p85_kmh": lanes.quantile(0.85).round(1), "min_kmh": lanes.min(), "max_kmh": lanes.max()})
    lane_summary.index = [f"Lane {i}" + (" (left, fast)" if i == 1 else " (right)" if i == 5 else "") for i in lane_summary.index]
    lane_summary.to_csv(OUT / "lane_summary.csv", index_label="lane")

    snapshots, lane_counts = draw_video(video, tracks, vehicles, road, fps, OUT / "annotated.mp4")
    # README animation: 6 s at 10 frames/s, 720 px wide, shared palette (keeps the file small)
    frames = [Image.fromarray(cv2.cvtColor(cv2.resize(s, (720, 405), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
              for s in snapshots[::2]]
    sample = Image.new("RGB", (720, 405 * 4))
    for k, fr in enumerate(frames[:: max(len(frames) // 4, 1)][:4]):
        sample.paste(fr, (0, 405 * k))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)   # one palette from several frames
    pal = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]
    pal[0].save(ASSETS / "demo.gif", save_all=True, append_images=pal[1:], duration=150, loop=0, optimize=True)
    cv2.imwrite(str(ASSETS / "annotated_frame.jpg"), snapshots[len(snapshots) // 2], [cv2.IMWRITE_JPEG_QUALITY, 88])

    # trajectories on the road plane
    fig, ax = plt.subplots(figsize=(5.4, 7))
    for e in LANE_EDGES:
        ax.axvline(e, color="#bbb", lw=1, ls="-" if e in (LANE_EDGES[0], LANE_EDGES[-1]) else "--")
    ax.axhspan(*ZONE, color="#1c69d4", alpha=0.05)
    keep = set(vehicles.track_id)
    for tid, g in tracks[tracks.track_id.isin(keep) & ~tracks.truncated].groupby("track_id"):
        g = g[(g.Y > -12) & (g.Y < 60)]
        lc = vehicles.set_index("track_id").lane_change.get(tid, False)
        ax.plot(g.X, g.Y, color="#d6336c" if lc else "#1c69d4", lw=1.6 if lc else 0.9, alpha=0.9 if lc else 0.6)
    ax.axhline(COUNT_LINE_Y, color="#333", lw=1)
    ax.text(LANE_EDGES[-1] + 0.2, COUNT_LINE_Y, "counting line", fontsize=8, va="center")
    for i in range(5):
        ax.text((LANE_EDGES[i] + LANE_EDGES[i + 1]) / 2, -11, f"L{i + 1}", ha="center", fontsize=9)
    ax.set(xlim=(LANE_EDGES[0] - 0.5, LANE_EDGES[-1] + 4), ylim=(-12, 60), xlabel="across the road (m)",
           ylabel="along the road (m)", title="Vehicle trajectories, top-down\n(red = lane change)")
    ax.set_aspect(0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "trajectories.png", dpi=150)

    # speed by lane
    fig, ax = plt.subplots(figsize=(7, 3.8))
    data = [vehicles[vehicles.lane == i].speed_kmh for i in range(1, 6)]
    ax.boxplot(data, tick_labels=[f"Lane {i}" for i in range(1, 6)], widths=0.5, showfliers=False,
               medianprops={"color": "#1c69d4", "lw": 2})
    for i, d in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(d)), d, s=14, color="#1c69d4", alpha=0.6)
    ax.set(ylabel="speed (km/h)", title=f"Measured speeds by lane ({len(vehicles)} vehicles, lane 1 = left)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "speed_by_lane.png", dpi=150)

    summary = {"tracker": args.tracker, "clip_seconds": round(n_frames / fps, 1), "fps": fps,
               "vehicles_with_speed": int(len(vehicles)), "vehicles_counted_at_line": int(sum(lane_counts.values())),
               "lane_changes": int(vehicles.lane_change.sum()),
               "speed_kmh": {"mean": round(float(vehicles.speed_kmh.mean()), 1),
                             "median": round(float(vehicles.speed_kmh.median()), 1),
                             "p85": round(float(vehicles.speed_kmh.quantile(0.85)), 1)},
               "median_speed_uncertainty_kmh": round(float(vehicles.speed_ci_kmh.median()), 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\n" + lane_summary.to_string())


if __name__ == "__main__":
    main()
