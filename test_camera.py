import cv2
import depthai as dai
from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager

pm = PipelineManager()
pm.createColorCam(xout=True)


with dai.Device(pm.pipeline) as device:
    pv = PreviewManager(display=[Previews.color.name])
    pv.createQueues(device)

    while True:
        pv.prepareFrames()
        pv.showFrames()

        if cv2.waitKey(1) == ord("q"):
            break
