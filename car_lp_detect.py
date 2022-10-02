#!/usr/bin/env python3

import argparse
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
from depthai_sdk import FPSHandler

from utils import frame_norm, send_frame_to_queue

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", default=True, help="Debug mode")
parser.add_argument("-cam", "--camera", action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument("-vid", "--video", type=argparse.FileType("r", encoding="UTF-8"), help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument("-nn", "--nn-blob-model", type=argparse.FileType("r", encoding="UTF-8"), help="Set path of the blob (NN model)")
parser.add_argument("-nnt", "--nn-threshold", type=float, default=0.5, help="Neural Networks Confidence Thresholds")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError('No source selected. Use either "-cam" to run on RGB camera as a source or "-vid <path>" to run on video')

VEH_NN_INPUT_IMG_WIDTH = 256
VEH_NN_INPUT_IMG_HEIGHT = 256

LP_NN_INPUT_IMG_WIDTH = 300
LP_NN_INPUT_IMG_HEIGHT = 300

SHAVES = 6 if args.camera else 8

pipeline = dai.Pipeline()

# cam/vid -> veh_manip -> veh_nn -> lp_nn

if args.camera:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1024, 768)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
else:
    # create a XLinkIn to send the video frames from video file
    vid = pipeline.create(dai.node.XLinkIn)
    vid.setStreamName("vid")
    vid.setNumFrames(4)
    # create a video file capture
    cap = cv2.VideoCapture(str(Path(args.video.name).resolve().absolute()))

# Vehicle detection NN
veh_nn = pipeline.createMobileNetDetectionNetwork()
veh_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-0200", shaves=SHAVES))
veh_nn.setConfidenceThreshold(args.nn_threshold)
veh_nn.setNumInferenceThreads(2)
veh_nn.input.setQueueSize(1)

# license plate detection NN
lp_nn = pipeline.createMobileNetDetectionNetwork()
lp_nn.setBlobPath(args.nn_blob_model.name)
lp_nn.setConfidenceThreshold(args.nn_threshold)
lp_nn.setNumInferenceThreads(2)
lp_nn.input.setQueueSize(1)

# ImageManip will resize the frame coming from the camera/video
# before sending it to the license plate detection NN node
veh_manip = pipeline.create(dai.node.ImageManip)
veh_manip.initialConfig.setResize(VEH_NN_INPUT_IMG_WIDTH, VEH_NN_INPUT_IMG_HEIGHT)
veh_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
veh_manip.out.link(veh_nn.input)

# use if we want to run the license plate NN in parallel with the veh NN
# lic_manip = pipeline.create(dai.node.ImageManip)
# lic_manip.initialConfig.setResize(LP_NN_INPUT_IMG_WIDTH, LP_NN_INPUT_IMG_HEIGHT)
# lic_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
# lic_manip.out.link(lp_nn.input)

# any frame sent to in_veh queue will have the dimensions and characteristics
# required by the license plate NN
# in_veh.out.link(lic_manip.input) # this is in case we will pre-process the image before
in_veh = pipeline.create(dai.node.XLinkIn)
in_veh.setStreamName("in_veh")
in_veh.out.link(lp_nn.input)


# Send cam/vid to the host
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Send detections to the host (for bounding boxes)
xout_veh_det = pipeline.create(dai.node.XLinkOut)
xout_veh_det.setStreamName("veh_det")

xout_lp_det = pipeline.create(dai.node.XLinkOut)
xout_lp_det.setStreamName("lp_det")

# connect detections to xout
veh_nn.out.link(xout_veh_det.input)
lp_nn.out.link(xout_lp_det.input)

# connect cam/vid to xout to see these as it is and
# send the source frames (cam/vid) to the image manipulation,
# to resize the image frame according to the NN
if args.camera:
    cam.preview.link(xout_rgb.input)
    cam.preview.link(veh_manip.inputImage)
else:
    vid.out.link(xout_rgb.input)
    vid.out.link(veh_manip.inputImage)

# to manage the frames rate
if args.camera:
    fps = FPSHandler()
else:
    fps = FPSHandler(cap)


def should_run() -> bool:
    """
    This is needed to validate if the video is
    loaded, for camera always is true
    """
    return cap.isOpened() if args.video else True


# def veh_thread(detect_queue: dai.DataOutputQueue, send_queue: dai.DataInputQueue) -> None:

#     while RUNNING:
#         try:
#             detections = []
#             in_queue = detect_queue.get()
#             if in_queue is not None:
#                 detections = in_queue.detections

#             for det in detections:
#                 bbox = frame_norm(orig_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

#         except RuntimeError:
#             continue


with dai.Device(pipeline) as device:
    RUNNING = True

    veh_det = device.getOutputQueue("veh_det", 1, False)  # to get the detection from vehicle NN
    lp_det = device.getOutputQueue("lp_det", 1, False)  # to get the detection from license plate NN
    q_rgb = device.getOutputQueue("rgb", 1, True)  # to get the frame processed

    q_veh = device.getInputQueue("in_veh")  # to send the frames cropped with vehicle detection box
    q_vid = device.getInputQueue("vid")  # to send the frames coming from video file

    veh_detections = []
    lp_detections = []

    if not args.camera:
        # size of frames coming from video file
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while should_run():
        if not args.camera:
            send_frame_to_queue(cap, q_vid, (video_width, video_height))

        frame = q_rgb.get().getCvFrame()
        # sequence = q_rgb.getSequenceNum()

        veh_det_data = veh_det.tryGet()
        if veh_det_data is not None:
            veh_detections = veh_det_data.detections

        fps.nextIter()
        if frame is not None:
            for veh in veh_detections:
                bbox = frame_norm(frame, (veh.xmin, veh.ymin, veh.xmax, veh.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color=(0, 255, 0))
            cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    RUNNING = False

print("FPS: {:.2f}".format(fps.fps()))
if not args.camera:
    cap.release()
