from utils.yolov8 import YOLOv8
from utils.wrappers import ModelWrapper
from utils.camera import setup_camera
import cv2

MODEL_PATH = './models/yolov8n-face.rknn'
CAM_WIDTH = 640
CAM_HEIGHT = 640

if __name__ == '__main__':
    model, _ = ModelWrapper.setup(MODEL_PATH)
    yolov8 = YOLOv8(model)
    capture = setup_camera(CAM_WIDTH, CAM_HEIGHT)

    while capture.isOpened():
        status, frame = capture.read()
        
        if not status:
            break

        boxes, scores, classids, kpts = yolov8.detect(frame)
        dstimg = yolov8.draw_detections(frame, boxes, scores, kpts)

        cv2.imshow('Webcam', frame)
        

    # release
    model.release()
