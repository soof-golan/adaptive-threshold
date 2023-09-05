import cv2
import numpy as np

from helpers import (
    cleanup,
    open_camera,
    q_key_pressed,
    stream,
    to_float,
    to_unit8,
)

SHOW_INTERMEDIATE = True


def adaptive_threshold(x: np.ndarray, block_size: int, threshold: float) -> np.ndarray:
    """Adaptive thresholding"""
    x = to_float(x)
    mean = cv2.GaussianBlur(
        x,
        (block_size, block_size),
        sigmaX=0.0,
        sigmaY=0.0,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED,
    )
    if SHOW_INTERMEDIATE:
        cv2.imshow("Gaussian Blur", mean)
    x = mean - x
    x = to_unit8(x)
    if SHOW_INTERMEDIATE:
        cv2.imshow("Difference", x)
    # x = cv2.threshold(x, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    x = np.where(x > threshold, 0, 255).astype(np.uint8)
    return to_unit8(x)


def demo_threshold(
    camera: cv2.VideoCapture,
    block_size: int = 11,
    pre_blur: int = 3,
    resize: float = 0.5,
    threshold: float = 2.0,
):
    for frame in stream(camera):
        frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if pre_blur > 0:
            gray = cv2.GaussianBlur(gray, (pre_blur, pre_blur), 0)
        if SHOW_INTERMEDIATE:
            cv2.imshow("Input", gray)
        ours = adaptive_threshold(gray, block_size, threshold)

        cv2.imshow("Gaussian Adaptive Threshold", ours)

        if q_key_pressed():
            break


def classic_adaptive_threshold(block_size, gray, threshold):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        threshold,
    )


def main():
    device = 0
    with open_camera(device) as camera:
        demo_threshold(camera)

    cleanup()


if __name__ == "__main__":
    main()
