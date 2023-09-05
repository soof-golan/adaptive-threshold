import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager

import cv2
import numpy as np


@contextlib.contextmanager
def open_camera(device_index: int) -> AbstractContextManager[cv2.VideoCapture]:
    cap = cv2.VideoCapture(device_index)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        yield cap
    finally:
        cap.release()


def stream(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        yield frame


@contextlib.contextmanager
def window(name: str) -> AbstractContextManager:
    cv2.namedWindow(
        name,
        cv2.WINDOW_AUTOSIZE
        | cv2.WINDOW_GUI_NORMAL
        | cv2.WINDOW_KEEPRATIO
        | cv2.WINDOW_OPENGL,
    )
    try:
        yield
    finally:
        cv2.destroyWindow(name)


def cleanup():
    cv2.destroyAllWindows()


def q_key_pressed():
    return cv2.waitKey(1) & 0xFF == ord("q")


def to_float(x: np.ndarray) -> np.ndarray:
    """Converts an image to float32"""
    if x.dtype == np.float32:
        return x.clip(0.0, 1.0)
    elif x.dtype == np.float64:
        return x.astype(np.float32).clip(0.0, 1.0)
    elif x.dtype == np.uint8:
        return (x.astype(np.float32) * (1.0 / 255.0)).clip(0.0, 1.0)
    elif x.dtype == np.uint16:
        return (x.astype(np.float32) * (1.0 / 65535.0)).clip(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")


def to_unit8(x: np.ndarray) -> np.ndarray:
    """Converts an image to uint8"""
    if x.dtype == np.uint8:
        return x
    elif x.dtype == np.bool_:
        return x.astype(np.uint8) * 255
    elif x.dtype == np.float32:
        return (x.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    elif x.dtype == np.float64:
        return (x.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    elif x.dtype == np.uint16:
        return (x.clip(0.0, 1.0) * 65535.0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")
