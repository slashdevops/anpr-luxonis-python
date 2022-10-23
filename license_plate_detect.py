#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", default=True, help="Debug mode")
parser.add_argument("-cam", "--camera", action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument("-vid", "--video", type=argparse.FileType("r", encoding="UTF-8"), help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument("-nn", "--nn-blob-model", type=argparse.FileType("r", encoding="UTF-8"), help="Set path of the blob (NN model)")
parser.add_argument("-nnt", "--nn-threshold", type=float, default=0.5, help="Neural Network Confidence Thresholds")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError('No source selected. Use either "-cam" to run on RGB camera as a source or "-vid <path>" to run on video')

NN_INPUT_IMG_WIDTH = 300
NN_INPUT_IMG_HEIGHT = 300

SHAVES = 6 if args.camera else 8

pipeline = dai.Pipeline()

if args.camera:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(800, 600)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
else:
    # create a XLinkIn to send the video frames
    vid = pipeline.create(dai.node.XLinkIn)
    vid.setStreamName("vid")
    cap = cv2.VideoCapture(str(Path(args.video.name).resolve().absolute()))

# NN
lp_nn = pipeline.createMobileNetDetectionNetwork()
lp_nn.setBlobPath(args.nn_blob_model.name)
lp_nn.setConfidenceThreshold(args.nn_threshold)
lp_nn.setNumInferenceThreads(2)
lp_nn.input.setQueueSize(1)

# ImageManip will resize the frame coming from the camera
# before sending it to the license plate detection NN node
lp_manip = pipeline.create(dai.node.ImageManip)
lp_manip.initialConfig.setResize(NN_INPUT_IMG_WIDTH, NN_INPUT_IMG_HEIGHT)
lp_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
lp_manip.out.link(lp_nn.input)

# send the source in frames to the image manipulation
if args.camera:
    cam.preview.link(lp_manip.inputImage)  # send camera frames to imageManip node
else:
    vid.out.link(lp_nn.input)


# Send video or cam to the host
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Send detections to the host (for bounding boxes)
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detection")

# connect detections to xout
lp_nn.out.link(xout_nn.input)

# connect cam/vid to xout
if args.camera:
    cam.preview.link(xout_rgb.input)

# to manage the frames
if args.camera:
    fps = FPSHandler()
else:
    fps = FPSHandler(cap)


def frame_norm(frame: object, bbox: tuple) -> np.ndarray:
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def get_frame() -> tuple[bool, object]:
    if args.camera:
        q_rgb = device.getOutputQueue("rgb", 1, True)
        return True, q_rgb.get().getCvFrame()

    return cap.read()


def should_run() -> bool:
    """
    This is needed to validate if the video is
    loaded, for camera always is true
    """
    return cap.isOpened() if args.video else True


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def to_depthai_frame(frame: object, size: tuple) -> dai.ImgFrame:
    time_stamp = time.monotonic()
    img = dai.ImgFrame()
    img.setData(to_planar(frame, size))
    img.setTimestamp(time_stamp)
    img.setType(dai.RawImgFrame.Type.BGR888p)
    img.setWidth(size[0])
    img.setHeight(size[1])

    return img


with dai.Device(pipeline) as device:
    q_nn = device.getOutputQueue("detection", 1, False)

    detections = []

    while should_run():
        ok, frame = get_frame()
        if not ok:
            break

        # send the video frames to cam processor
        if not args.camera:
            q_vid = device.getInputQueue("vid", 1, False)
            img_frame = to_depthai_frame(frame, (NN_INPUT_IMG_WIDTH, NN_INPUT_IMG_HEIGHT))
            q_vid.send(img_frame)

        in_nn = q_nn.tryGet()
        if in_nn is not None:
            detections = in_nn.detections

        fps.nextIter()
        if frame is not None:
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.putText(frame, f"Fps: {fps.fps():.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color=(0, 255, 0))
            cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break
