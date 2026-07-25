# Plant Vision Browser

Captures the Pi Camera Module 2 through `libcamerasrc`, detects ArUco markers 1–3, and serves annotated video in a browser.

## Dependencies

```bash
sudo apt update
sudo apt install python3-opencv python3-flask python3-numpy python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-libcamera
```

## Generate markers

```bash
cd ~/plant_vision_browser
python3 generate_markers.py
```

## Run

```bash
python3 app.py
```

Find the Pi IP:

```bash
hostname -I
```

Open `http://PI_IP:5000` from a device on the same network.
