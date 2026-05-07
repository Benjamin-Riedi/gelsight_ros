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


def _fourcc_to_str(fourcc_code: Union[int, float]) -> str:
    """Convert OpenCV FOURCC float value to printable 4-character string."""
    try:
        value = int(fourcc_code)
        chars = []
        for i in range(4):
            c = chr((value >> (8 * i)) & 0xFF)
            chars.append(c if 32 <= ord(c) <= 126 else "?")
        return "".join(chars)
    except (TypeError, ValueError, OverflowError):
        return "????"


def _backend_flag_to_name(backend_flag: Optional[int]) -> str:
    if backend_flag is None:
        return "default"
    if backend_flag == cv2.CAP_V4L2:
        return "v4l2"
    if backend_flag == cv2.CAP_GSTREAMER:
        return "gstreamer"
    return str(backend_flag)


class GelSightMiniRGBCompat:
    """RGB-only OpenCV capture helper, compatible with Python 3.8+."""

    def __init__(
        self,
        target_width=640,
        target_height=480,
        border_fraction=0.15,
        prefer_v4l2=True,
        backend: Optional[str] = None,
        buffersize: Optional[int] = 1,
        fps: Optional[float] = 25.0,
        fourcc: Optional[str] = None,
        warmup_grabs: int = 3,
        log_capture_properties: bool = True,
    ):
        """Initialize camera helper.

        backend values (Linux only): "default"/"auto", "v4l2", "gstreamer".
        Set buffersize=None or fps=None to skip those property requests.
        """
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.border_fraction = float(border_fraction)
        self.prefer_v4l2 = bool(prefer_v4l2)
        self.backend = backend
        # Set to None to skip requesting these properties on open().
        self.buffersize = None if buffersize is None else int(buffersize)
        self.requested_fps = None if fps is None else float(fps)
        self.fourcc = fourcc
        self.warmup_grabs = max(0, int(warmup_grabs))
        self.log_capture_properties = bool(log_capture_properties)
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

    def _resolve_backend_flag(self):
        if platform.system() != "Linux":
            return None

        backend = self.backend
        if backend is not None:
            backend = str(backend).strip().lower()
            if backend in ("default", "auto", ""):
                return None
            if backend == "v4l2":
                return cv2.CAP_V4L2
            if backend == "gstreamer":
                return cv2.CAP_GSTREAMER
            print(
                "Warning: unknown backend '{0}', using default. "
                "Valid options: default, auto, v4l2, gstreamer.".format(backend)
            )
            return None

        # Backward-compatible behavior when backend is not explicitly provided.
        if self.prefer_v4l2:
            return cv2.CAP_V4L2
        return None

    def open(self, device: Optional[DeviceRef] = None) -> None:
        resolved_device = self._resolve_device(device)

        if self.cap is not None:
            self.release()

        backend_flag = self._resolve_backend_flag()
        if backend_flag is None:
            self.cap = cv2.VideoCapture(resolved_device)
        else:
            self.cap = cv2.VideoCapture(resolved_device, backend_flag)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device: {resolved_device}")

        if self.buffersize is not None:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.buffersize))

        if self.requested_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(self.requested_fps))

        if self.fourcc:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(self.fourcc)))
            except (TypeError, ValueError, cv2.error) as exc:
                print(
                    "Warning: failed to set requested FOURCC '{0}': {1}".format(
                        self.fourcc, exc
                    )
                )

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

        if self.log_capture_properties:
            actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            actual_fourcc = _fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC))
            backend_prop = getattr(cv2, "CAP_PROP_BACKEND", None)
            actual_backend = (
                int(self.cap.get(backend_prop))
                if backend_prop is not None
                else None
            )
            actual_buffersize = float(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
            backend_label = _backend_flag_to_name(backend_flag)
            print(
                "Capture properties: backend_flag={0}, backend={1}, device={2}, "
                "size={3}x{4}, fps={5:.2f}, fourcc='{6}', buffersize={7}".format(
                    backend_label,
                    actual_backend,
                    resolved_device,
                    actual_width,
                    actual_height,
                    actual_fps,
                    actual_fourcc,
                    actual_buffersize,
                )
            )

        for i in range(self.warmup_grabs):
            if not self.cap.grab():
                print(
                    "Warning: warmup grab failed at attempt {0} of {1}.".format(
                        i + 1, self.warmup_grabs
                    )
                )
                break

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
