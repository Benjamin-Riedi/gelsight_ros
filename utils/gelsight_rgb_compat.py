import glob
import os
import platform
import re
import time
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


DeviceRef = Union[int, str]


def _crop_and_resize(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    border_fraction: float = 0.15,
) -> np.ndarray:
    # Clamp below 0.5 so cropping never removes all rows/columns.
    border_fraction = min(max(0.0, border_fraction), 0.49)
    border_rows = int(image.shape[0] * border_fraction)
    border_cols = int(image.shape[1] * border_fraction)

    cropped = image[
        border_rows : image.shape[0] - border_rows,
        border_cols : image.shape[1] - border_cols,
    ]

    if target_size is not None:
        cropped = cv2.resize(cropped, target_size)

    return cropped


class GelSightMiniRGBCompat:
    """RGB-only OpenCV capture helper, compatible with Python 3.8+."""

    def __init__(
        self,
        target_width=640,
        target_height=480,
        border_fraction=0.15,
        prefer_v4l2=True,
    ):
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.border_fraction = float(border_fraction)
        self.prefer_v4l2 = bool(prefer_v4l2)
        self.cap = None
        self.fps = 0.0
        self._time_prev = time.time()

    @staticmethod
    def list_devices() -> Dict[int, str]:
        devices = {}

        if platform.system() == "Linux":
            by_id = sorted(glob.glob("/dev/v4l/by-id/*"))
            if by_id:
                for idx, path in enumerate(by_id):
                    devices[idx] = path
                return devices

            video_nodes = sorted(glob.glob("/dev/video*"))
            for path in video_nodes:
                match = re.search(r"/dev/video(\\d+)$", path)
                if match:
                    devices[int(match.group(1))] = path
            return devices

        for idx in range(0, 10):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                devices[idx] = "Video Device {0}".format(idx)
                cap.release()

        return devices

    def _resolve_device(self, device: Optional[DeviceRef]) -> DeviceRef:
        if device is not None:
            return device

        devices = self.list_devices()
        if not devices:
            return 0

        return devices[min(devices.keys())]

    def open(self, device: Optional[DeviceRef] = None) -> None:
        resolved_device = self._resolve_device(device)

        if self.cap is not None:
            self.release()

        if platform.system() == "Linux" and self.prefer_v4l2:
            self.cap = cv2.VideoCapture(resolved_device, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(resolved_device)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device: {resolved_device}")

        width_ok = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.target_width))
        height_ok = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.target_height))
        if not width_ok or not height_ok:
            print(
                "Warning: camera driver did not confirm requested frame size properties."
            )
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != self.target_width or actual_height != self.target_height:
            print(
                "Warning: requested resolution {0}x{1}, got {2}x{3}".format(
                    self.target_width,
                    self.target_height,
                    actual_width,
                    actual_height,
                )
            )
        self._time_prev = time.time()

    def read_rgb(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera is not opened.")

        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera.")

        now = time.time()
        dt = now - self._time_prev
        self.fps = 1.0 / dt if dt > 0.0 else 0.0
        self._time_prev = now

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = _crop_and_resize(
            frame_rgb,
            target_size=(self.target_width, self.target_height),
            border_fraction=self.border_fraction,
        )
        return frame_rgb

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
