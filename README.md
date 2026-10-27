# 🛣️ Vehicle Tracking, Trajectories & Speed Estimation

Track every vehicle seen by a fixed traffic camera, follow its trajectory on the road in **metres**, measure its **speed in km/h**, and turn the result into traffic statistics: counts and flow per lane, 85th-percentile speeds and lane changes. This is the camera-based measurement behind traffic monitoring, smart-infrastructure (V2X) sensing and the reference data used to validate automated-driving functions.

![Demo](assets/demo.gif)
<sub>Real freeway footage from an overpass. Box colour = speed (green slower, red faster). Speeds are shown live inside the calibrated zone and afterwards as the vehicle's measured speed. Top right: bird's-eye view of the first 60 m.</sub>

Part of a portfolio series with [Automotive Object Detection](https://github.com/souadessemyhij1999/automotive-object-detection) · [Road Segmentation](https://github.com/souadessemyhij1999/road-scene-segmentation) · [Driving Perception System](https://github.com/souadessemyhij1999/driving-perception-system) (tracking from inside the car). This project tracks from the **infrastructure side**.

---

## How it works

```
 video frame ──► YOLO26-s detector (vehicles only, run once and cached)
                     │
                     ▼
               tracker: Centroid (classic OpenCV)  |  ByteTrack  |  BoT-SORT
                     │   identity from the tracker, position from the raw detection
                     ▼
         box bottom-centre (where the rear meets the road)
                     │   homography calibrated from lane markings
                     ▼
         road coordinates in metres ─► robust speed fit (Theil-Sen) inside the calibrated zone
                                    ─► lane, lane changes, counting line, trajectories
```

| Step | What I did and why |
|---|---|
| **Empty road** | The per-pixel median of 82 frames removes all moving vehicles. It's used for calibration and the test bench. |
| **Calibration from lane markings** | US interstate markings are standardised (MUTCD): 3.05 m dashes every 12.19 m, 3.66 m lanes. Dashes are found automatically and their endpoints fit a pixel→metre homography. |
| **Speed** | Slope of a robust straight-line fit of distance-along-road over time, using only the zone where the calibration was checked (−8 m to +30 m). |
| **Lane changes** | A lane change needs a sideways move of at least 2.5 m and a different lane at the start and end, judged on short smoothed windows. |
| **Test bench with ground truth** | Real traffic video has no "true" speeds or IDs to score against, so I built one: the empty road, with real vehicle cut-outs replayed at **known speeds, lanes and lane changes**, rendered with the calibrated perspective. This is the same idea as simulation-based testing in automotive validation. |

---

## Results

### 1. Calibration: checked on lane markings the fit never saw

Fit on the 2 nearest rows of dashes (16 endpoints), then tested on the next row, 24–27 m away:

| Held-out check | Measured | Standard | Error |
|---|---|---|---|
| Dash length | 3.06 m | 3.05 m | **≤ 7 cm** |
| Lane width | 3.62 m | 3.66 m | ≤ 10 cm |
| Dash position along the road | −0.45 m off at 24 m | – | ≈ 1.8% |

![Calibration](assets/calibration.png)
<sub>Yellow: metric grid (3.66 m lanes × 10 m) projected back into the image. Green: dashes used for the fit. Red: held-out dashes.</sub>

### 2. Test bench: trackers scored against ground truth

25 s, 38 vehicles, 9 lane changes, true speeds 84–119 km/h. Standard multi-object-tracking metrics ([py-motmetrics](https://github.com/cheind/py-motmetrics)) in the zone up to 50 m:

| Tracker | MOTA | IDF1 | ID switches | Vehicles counted (true 38) | Speed error (mean abs.) | Speed bias | Lane changes found (of 9) |
|---|---|---|---|---|---|---|---|
| Centroid (classic OpenCV) | 0.951 | 0.866 | 34 | **41** ✗ | 0.18 km/h | −0.09 | 8 |
| ByteTrack, Kalman-smoothed boxes | 0.821 | 0.908 | 3 | 38 ✓ | 0.98 km/h | **+0.98** | 9 |
| ByteTrack | 0.947 | 0.971 | 3 | 38 ✓ | 0.13 km/h | −0.08 | 9 |
| BoT-SORT, Kalman-smoothed boxes | 0.952 | 0.975 | 1 | 38 ✓ | 0.97 km/h | **+0.97** | 9 |
| **BoT-SORT** (used) | **0.949** | **0.974** | **1** | **38** ✓ | **0.14 km/h** | −0.08 | **9, 0 false** |

![Bench speed accuracy](assets/bench_speed_accuracy.png)

Detection recall was 99.8–100% from 0 to 50 m, and 92.5% in the first metres, where vehicles are cut off by the bottom of the image.

### 3. Real freeway clip: traffic analytics

33 seconds, **54 vehicles** measured, all counted at the counting line.

| Lane | Vehicles | Flow (veh/h)* | Mean speed | Median | **85th percentile** |
|---|---|---|---|---|---|
| Lane 1 (left, fast) | 5 | 548 | 96.4 km/h | 99.0 | 99.7 |
| Lane 2 | 14 | 1,535 | 89.5 km/h | 89.9 | 95.3 |
| Lane 3 | 13 | 1,425 | 84.9 km/h | 84.9 | 88.9 |
| Lane 4 | 19 | 2,083 | 77.2 km/h | 77.6 | 79.8 |
| Lane 5 (right) | 3 | 329 | 72.2 km/h | 70.3 | 74.3 |
| **All** | **54** | | **83.7 km/h** | **83.3** | **93.0** |

<sub>*Flow extrapolated from 33 seconds, so it indicates the order of magnitude only. The 85th-percentile speed is the standard traffic-engineering figure used for setting speed limits.</sub>

| Trajectories on the road (top-down) | Speeds by lane |
|---|---|
| ![Trajectories](assets/trajectories.png) | ![Speed by lane](assets/speed_by_lane.png) |

---

## Key observations

- **Identity matters more than position for traffic counts.** The classic centroid tracker places boxes well (good MOTA, accurate speeds) but switched IDs 34 times and **counted 41 vehicles instead of 38**, an 8% over-count. BoT-SORT counted exactly 38 with one ID switch. For a traffic authority, the count is the product.
- **A subtle bias found by the test bench.** Using the trackers' own Kalman-smoothed boxes made every speed read **about 1 km/h too fast**. Their motion model assumes constant speed *in pixels*, but a car driving away slows down in the image as perspective shrinks it, so the filter overshoots. Taking identity from the tracker but position from the raw detection removed the bias (+0.98 → −0.08 km/h) and lifted ByteTrack's MOTA from 0.82 to 0.95.
- **The bench also caught a flawed lane-change rule.** The first version compared one-second windows, but at 30 m/s one second covers 30 m of road, so the manoeuvre got averaged away. It **found only 1 of 9** lane changes. Short 0.3 s windows on a smoothed path found **9 of 9 with no false alarms**.
- **Lane changes need a physical threshold.** A looser rule first flagged 3 "lane changes" on the real clip that were cars drifting about 1 m along a lane line. With the 2.5 m rule the real clip shows none: the largest sideways move of any vehicle was 1.8 m.
- **The traffic pattern is plausible.** Speeds fall steadily from the fast lane (96 km/h) to the right lane (72 km/h), and the busiest lane is lane 4. That isn't proof, but it is the pattern expected on a real freeway.
- **Real-world speed accuracy is limited by calibration, not tracking.** On the bench the geometry is perfect, so errors are tiny (0.14 km/h). On the real road, the held-out dash row sits about 1.8% short, so real speeds carry an estimated **±2% (≈ ±2 km/h)** uncertainty, more than 10× the tracking error. Better calibration would be the next investment.

---

## Project structure

```
traffic-speed-tracking/
├── configs/road_homography.yaml   # pixel -> metre mapping + held-out validation results
├── src/
│   ├── data.py                    # downloads the freeway clip
│   ├── calibrate.py               # empty road + lane-dash detection + homography + validation
│   ├── tracking.py                # cached detection + Centroid / ByteTrack / BoT-SORT
│   ├── road.py                    # metres, lanes, speed fit, lane-change rule
│   ├── bench.py                   # semi-synthetic test video with exact ground truth
│   ├── evaluate_bench.py          # MOTA / IDF1 / counts / speed / lane-change scores
│   └── analyze_freeway.py         # real clip: annotated video, analytics, charts
├── results/
│   ├── bench/                     # ground truth, tracker comparison, speed errors, lane changes
│   └── freeway/                   # vehicles.csv, lane_summary.csv, summary.json
└── assets/                        # README images and demo animation
```

## Quick start

```bash
pip install -r requirements.txt
python src/data.py               # download the freeway clip
python src/calibrate.py          # homography from lane markings
python src/analyze_freeway.py    # track, measure speeds, analytics (detection ~8 min on a laptop CPU)
python src/bench.py              # build the test bench
python src/evaluate_bench.py     # score the trackers against ground truth
```

---

## Limitations & next steps

- **The calibration is validated up to about 27 m**, since farther dashes are only a few pixels long. Speeds are measured inside that zone, and lateral positions drift slightly beyond 45 m on the right-hand lanes.
- **The test bench is cleaner than reality.** Constant speeds, perfect geometry and pasted vehicles mean the bench measures tracking and measurement logic, not calibration error, weather or night conditions.
- **Short clip.** 33 seconds gives counts and speeds for 54 vehicles, not representative traffic volumes.
- **Next:** calibrate with more reference points (or a surveyed one), add night and rain footage, export vehicle data in real time (for example as MQTT messages for a V2X roadside unit), and validate speeds against radar ground truth such as the BrnoCompSpeed dataset.

## Tech stack

Python · Ultralytics YOLO26 · ByteTrack · BoT-SORT · OpenCV (homography, morphology, connected components) · SciPy (Theil-Sen, Hungarian matching) · py-motmetrics · pandas · matplotlib

## Credits & licences

- Detector and trackers: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Freeway clip: `CarsDrivingUnderBridge.mp4` from [MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp](https://github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp). No licence is stated by its source, so the video is downloaded at runtime and not stored in this repository
- Lane-marking dimensions: US Federal Highway Administration, Manual on Uniform Traffic Control Devices (MUTCD)
- Project idea from [project-based-learning](https://github.com/practical-tutorials/project-based-learning)

Released under AGPL-3.0 to match Ultralytics.

**Author:** Souad Essemyhij, MSc Robotics and Autonomous Systems, Universität zu Lübeck · [LinkedIn](https://www.linkedin.com/in/souad-essemyhi-742a9520b/) · [GitHub](https://github.com/souadessemyhij1999)
