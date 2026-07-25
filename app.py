#!/usr/bin/env python3

import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from gi.repository import Gst, GLib

import serial

CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FPS = 15
DETECTION_INTERVAL_FRAMES = 3
LOW_LIGHT_THRESHOLD = 20
LOW_LIGHT_CONFIRMATIONS = 5
NORMAL_LIGHT_CONFIRMATIONS = 5
PRESENT_CONFIRMATIONS = 3
ABSENT_TIMEOUT_SECONDS = 3.0
EXPECTED_MARKER_IDS = (1, 2, 3)

GSTREAMER_PIPELINE = (
    f"libcamerasrc ! "
    f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 ! "
    f"videoconvert ! video/x-raw,format=BGR ! "
    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
)

@dataclass
class MarkerState:
    marker_id: int
    status: str = "UNKNOWN"
    consecutive_detections: int = 0
    last_seen_monotonic: float = 0.0
    last_seen_wall_time: Optional[str] = None
    centre_x: Optional[int] = None
    centre_y: Optional[int] = None

class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.average_brightness = 0.0
        self.camera_status = "STARTING"
        self.markers: Dict[int, MarkerState] = {
            marker_id: MarkerState(marker_id=marker_id)
            for marker_id in EXPECTED_MARKER_IDS
        }
        self.temperature = None
        self.humidity = None
        self.light = None

        self.soil = {
            1: None,
            2: None,
            3: None
        }

shared = SharedState()
app = Flask(__name__)
Gst.init(None)

aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
frame_number = 0
low_light_count = 0
normal_light_count = 0
stable_low_light = False

def wall_time_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def update_marker_states(detected_ids, centres, image_is_dark):
    now = time.monotonic()
    with shared.lock:
        for marker_id, marker in shared.markers.items():
            if image_is_dark:
                marker.status = "UNKNOWN"
                marker.consecutive_detections = 0
                continue
            if marker_id in detected_ids:
                marker.consecutive_detections += 1
                marker.last_seen_monotonic = now
                marker.last_seen_wall_time = wall_time_string()
                marker.centre_x, marker.centre_y = centres[marker_id]
                if marker.consecutive_detections >= PRESENT_CONFIRMATIONS:
                    marker.status = "PRESENT"
            else:
                marker.consecutive_detections = 0
                if marker.last_seen_monotonic == 0.0 or now - marker.last_seen_monotonic >= ABSENT_TIMEOUT_SECONDS:
                    marker.status = "ABSENT"

def draw_status_panel(frame, brightness):
    with shared.lock:
        camera_status = shared.camera_status

        marker_snapshots = [
            MarkerState(**asdict(marker))
            for marker in shared.markers.values()
        ]

        temperature = shared.temperature
        humidity = shared.humidity
        light = shared.light

        soil_1 = shared.soil[1]
        soil_2 = shared.soil[2]
        soil_3 = shared.soil[3]

    def display_value(value, suffix):
        if value is None:
            return "Unavailable"

        return f"{value:.1f}{suffix}"

    cv2.rectangle(
        frame,
        (8, 8),
        (385, 300),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Camera: {camera_status}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Brightness: {brightness:.1f}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y = 84

    for marker in marker_snapshots:
        cv2.putText(
            frame,
            f"Pot {marker.marker_id}: {marker.status}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        y += 24

    cv2.putText(
        frame,
        f"Temperature: {display_value(temperature, ' C')}",
        (18, 166),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Humidity: {display_value(humidity, ' %')}",
        (18, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Light: {display_value(light, ' lux')}",
        (18, 214),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Soil 1: {display_value(soil_1, ' %')}",
        (18, 238),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Soil 2: {display_value(soil_2, ' %')}",
        (18, 262),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Soil 3: {display_value(soil_3, ' %')}",
        (18, 286),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

def process_frame(frame):
    global frame_number
    global low_light_count
    global normal_light_count
    global stable_low_light
    frame_number += 1
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(grayscale))

    raw_low_light = brightness < LOW_LIGHT_THRESHOLD

    if raw_low_light:
        low_light_count += 1
        normal_light_count = 0

        if low_light_count >= LOW_LIGHT_CONFIRMATIONS:
            stable_low_light = True
    else:
        normal_light_count += 1
        low_light_count = 0

        if normal_light_count >= NORMAL_LIGHT_CONFIRMATIONS:
            stable_low_light = False

    image_is_dark = stable_low_light

    with shared.lock:
        shared.average_brightness = brightness
        shared.camera_status = "LOW LIGHT" if image_is_dark else "OK"

    detected_ids = set()
    centres = {}
    detection_frame = frame_number % DETECTION_INTERVAL_FRAMES == 0
    if not image_is_dark and detection_frame:
        corners, ids, _ = aruco_detector.detectMarkers(grayscale)
        if ids is not None:
            for marker_corners, marker_id_raw in zip(corners, ids.flatten()):
                marker_id = int(marker_id_raw)
                if marker_id not in EXPECTED_MARKER_IDS:
                    continue
                points = marker_corners.reshape((4,2)).astype(int)
                cv2.polylines(frame, [points], True, (0,255,0), 3)
                centre_x = int(points[:,0].mean())
                centre_y = int(points[:,1].mean())
                detected_ids.add(marker_id)
                centres[marker_id] = (centre_x, centre_y)
                cv2.circle(frame, (centre_x, centre_y), 5, (0,0,255), -1)
                top_left = points[0]
                cv2.putText(frame, f"Pot {marker_id}", (top_left[0], max(30, top_left[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    if image_is_dark or detection_frame:
        update_marker_states(detected_ids,centres,image_is_dark,)

    if image_is_dark:
        cv2.putText(frame, "LOW LIGHT - VISION UNKNOWN", (90, CAMERA_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
    draw_status_panel(frame, brightness)
    return frame

def on_new_sample(appsink):
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR
    try:
        frame = np.frombuffer(map_info.data, dtype=np.uint8)
        expected_size = width * height * 3
        if frame.size != expected_size:
            print(f"Unexpected frame size: got {frame.size}, expected {expected_size}")
            return Gst.FlowReturn.ERROR
        frame = frame.reshape((height, width, 3)).copy()
        annotated = process_frame(frame)
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with shared.lock:
                shared.latest_jpeg = encoded.tobytes()
    finally:
        buffer.unmap(map_info)
    return Gst.FlowReturn.OK

def camera_thread():
    pipeline = Gst.parse_launch(GSTREAMER_PIPELINE)
    appsink = pipeline.get_by_name("appsink")
    if appsink is None:
        raise RuntimeError("Could not find appsink in GStreamer pipeline.")
    appsink.connect("new-sample", on_new_sample)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    def on_bus_message(_, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"GStreamer error: {error}")
            print(f"Debug details: {debug}")
            with shared.lock:
                shared.camera_status = "ERROR"
    bus.connect("message", on_bus_message)
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Could not start GStreamer camera pipeline.")
    print("Camera pipeline started.")
    loop = GLib.MainLoop()
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

def uart_thread():

    uart = serial.Serial(
        "/dev/serial0",
        115200,
        timeout=1
    )

    print("UART thread started")

    while True:

        raw = uart.readline()

        if not raw:
            continue

        try:
            line = raw.decode().strip()
        except UnicodeDecodeError:
            continue

        parts = line.split(",")

        if len(parts) == 3 and parts[0] == "SENSOR":

            sensor = int(parts[1])
            value = float(parts[2])

            with shared.lock:
                shared.soil[sensor] = value

        elif len(parts) == 4 and parts[0] == "CLIMATE":

            with shared.lock:
                shared.temperature = float(parts[1])
                shared.humidity = float(parts[2])
                shared.light = float(parts[3])

def mjpeg_generator():
    while True:
        with shared.lock:
            jpeg = shared.latest_jpeg
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / CAMERA_FPS)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with shared.lock:
        return jsonify({
            "camera_status": shared.camera_status,
            "average_brightness": round(
                shared.average_brightness,
                1
            ),

            "markers": {
                marker_id: {
                    "status": marker.status,
                    "last_seen": marker.last_seen_wall_time,
                    "centre_x": marker.centre_x,
                    "centre_y": marker.centre_y,
                }
                for marker_id, marker
                in shared.markers.items()
            },

            "temperature": shared.temperature,
            "humidity": shared.humidity,
            "light": shared.light,

            "soil": {
                "1": shared.soil[1],
                "2": shared.soil[2],
                "3": shared.soil[3],
            },
        })

def main():
    camera = threading.Thread(
        target=camera_thread,
        daemon=True,
    )

    uart = threading.Thread(
        target=uart_thread,
        daemon=True,
    )

    camera.start()
    uart.start()

    print("Open http://<Raspberry-Pi-IP>:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False,
    )

if __name__ == "__main__":
    main()
