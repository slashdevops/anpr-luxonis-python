import time

import cv2
import depthai as dai
import numpy as np


def frame_norm(frame: object, bbox: tuple) -> np.ndarray:
    """
    This create a norm

    Parameters
    ----------
    frame: np.ndarray
        Object created with cv2.VideoCapture().read()

    bbox: tuple
        Tuple


    Returns
    -------
    np.ndarray
        Numpy NDArray object that represent the transformation of the arr after
        apply shape
    """
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Transform a np.ndarray (cv2.VideoCapture().read())

    Parameters
    ----------
    frame: np.ndarray
        Object created with cv2.VideoCapture().read()

    shape: tuple
        Tuple


    Returns
    -------
    np.ndarray
        Numpy NDArray object that represent the transformation of the arr after
        apply shape
    """
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def to_depthai_frame(frame: np.ndarray, size: tuple) -> dai.ImgFrame:
    """
    Transform a np.ndarray (cv2.VideoCapture().read()) frame to a depthai.ImgFrame
    changing it to BGR888p format given the size tuple represented by (width,height)

    Parameters
    ----------
    frame: np.ndarray
        Object created with cv2.VideoCapture().read()

    size: tuple
        Tuple with the width and heigh used to create the depthai.ImgFrame
    """
    time_stamp = time.monotonic()
    img = dai.ImgFrame()
    img.setData(to_planar(frame, size))
    img.setTimestamp(time_stamp)
    img.setType(dai.RawImgFrame.Type.BGR888p)
    img.setWidth(size[0])
    img.setHeight(size[1])

    return img


def send_frame_to_queue(video_capture: np.ndarray, send_queue: dai.DataInputQueue, size: tuple) -> None:
    """
    Send frame read from cv2.VideoCapture() to a dai.DataInputQueue after transform this to a depthai.ImgFrame()

    Parameters
    ----------
    video_capture: np.ndarray
        Object created with cv2.VideoCapture()

    send_queue: dai.DataInputQueue
        Object created with depthai.pipeline.create(dai.node.XLinkIn)

    size: tuple, required
        Tuple with the width and heigh used to create the depthai.ImgFrame
        before send this to send_queue
    """
    success, frame = video_capture.read()
    if success:
        img_frame = to_depthai_frame(frame, size)
        send_queue.send(img_frame)
