import glob
import os
import platform
import re
import time
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


DeviceRef = Union[int, str]

def detect_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=10,
        minRadius=100,
        maxRadius=160,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Detected circles: {circles}")
        return circles[0][0] # Return the first detected circle (x, y, radius)
    else:
        print("No circle detected.")
        return image

def crop_to_circle(image, circle, radius_factor=1.5):
    x, y, r = circle
    r = np.int16(np.around(r * radius_factor))

    # Crop bounding box around circle
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(image.shape[1], x + r)
    y2 = min(image.shape[0], y + r)

    crop = image[y1:y2, x1:x2]
    return crop



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


def _backend_flag_to_name(backend_flag: Optional[int]) -> str:
    if backend_flag is None:
        return "default"
    if backend_flag == cv2.CAP_V4L2:
        return "v4l2"
    if backend_flag == cv2.CAP_GSTREAMER:
        return "gstreamer"
    return str(backend_flag)


def _is_linux_video_device(device: DeviceRef) -> bool:
    if platform.system() != "Linux":
        return False
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        return device.startswith("/dev/video") or device.startswith("/dev/v4l/")
    return False


class GelSightMiniRGBCompat:
    """RGB-only OpenCV capture helper, compatible with Python 3.8+.

    Performance notes (Linux):
    - Backend, FOURCC, FPS and buffering have a huge impact on throughput.
    - For many UVC cameras, reducing CAP_PROP_BUFFERSIZE helps a lot.
    """

    def __init__(
        self,
        target_width=640,
        target_height=480,
        border_fraction=0.15,
        prefer_v4l2=True,
        backend: Optional[str] = None,
        fps: Optional[float] = 25.0,
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
        self.requested_fps = None if fps is None else float(fps)
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
                match = re.search(r"/dev/video(\d+)$", path)
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
        # return cv2.CAP_GSTREAMER
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

    def _read_capture_properties(self, backend_flag: Optional[int]):
        backend_prop = getattr(cv2, "CAP_PROP_BACKEND", None)
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "backend": int(self.cap.get(backend_prop)) if backend_prop is not None else None,
            "backend_label": _backend_flag_to_name(backend_flag),
        }

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

        width_ok = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.target_width))
        height_ok = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.target_height))
        if not width_ok or not height_ok:
            print(
                "Warning: camera driver did not confirm requested frame size properties."
            )
        props = self._read_capture_properties(backend_flag)
        actual_width = props["width"]
        actual_height = props["height"]
        actual_backend = props["backend"]
        backend_label = props["backend_label"]
        size_mismatch = (
            actual_width != self.target_width or actual_height != self.target_height
        )

        if size_mismatch:
            print(
                "Warning: requested resolution {0}x{1}, got {2}x{3}".format(
                    self.target_width,
                    self.target_height,
                    actual_width,
                    actual_height,
                )
            )

        if self.log_capture_properties:
            print(
                "Capture properties: backend_flag={0}, device={1}, "
                "size={2}x{3}".format(
                    backend_label,
                    resolved_device,
                    actual_width,
                    actual_height,
                )
            )

        uses_v4l2_backend = actual_backend == cv2.CAP_V4L2 or backend_label == "v4l2"
        if size_mismatch and _is_linux_video_device(resolved_device):
            if uses_v4l2_backend or backend_label == "default":
                print(
                    "Warning: Camera only supports {0}x{1} MJPG; decoding and resizing on CPU. "
                    "To improve performance, please ensure your camera/driver UVC supports "
                    "MJPG/YUYV at your desired resolution. "
                    "If using a recent kernel/firmware, try listing formats with "
                    "'v4l2-ctl --list-formats-ext' and upgrade drivers if needed.".format(
                        actual_width, actual_height
                    )
                )

        self._time_prev = time.time()

    def read_rgb(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera is not opened.")
        t_loop_start = time.perf_counter()
        ok, frame_bgr = self.cap.read()
        t1 = time.perf_counter()
        # print(f"Time to read frame: {(t1 - t_loop_start) * 1000:.2f} ms")
        if not ok:
            raise RuntimeError("Failed to read frame from camera.")

        now = time.time()
        dt = now - self._time_prev
        self.fps = 1.0 / dt if dt > 0.0 else 0.0
        self._time_prev = now
        # print(f"Effective capture FPS: {self.fps:.2f}")

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
