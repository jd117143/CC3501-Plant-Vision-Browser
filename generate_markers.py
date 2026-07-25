#!/usr/bin/env python3
from pathlib import Path
import cv2

OUTPUT_DIRECTORY = Path("markers")
MARKER_SIZE_PIXELS = 600
MARKER_IDS = (1, 2, 3)

def main():
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id in MARKER_IDS:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE_PIXELS)
        bordered = cv2.copyMakeBorder(marker, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
        output_path = OUTPUT_DIRECTORY / f"pot_{marker_id}.png"
        cv2.imwrite(str(output_path), bordered)
        print(f"Created {output_path}")

if __name__ == "__main__":
    main()
