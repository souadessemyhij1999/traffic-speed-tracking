"""
Download the freeway clip used in this project into data/ (not stored in this repository).

Source: "CarsDrivingUnderBridge.mp4" from github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp,
a 33-second, 1280x720, 30 fps clip filmed from a freeway overpass. The source repository states no
licence, so the video is downloaded from its original location rather than redistributed here.

    python src/data.py
"""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp/master/CarsDrivingUnderBridge.mp4"
DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA.mkdir(exist_ok=True)
    path = DATA / "freeway.mp4"
    if not path.exists():
        print("Downloading freeway clip (40 MB) ...")
        urllib.request.urlretrieve(URL, path)
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
