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

    while cv2.waitKey(1) < 0:
        status, frame = capture.read()
        
        if not status:
            break

        boxes, scores, classids, kpts = yolov8.detect(frame)
        idx = -1
        max_size = 0
        for i, box in enumerate(boxes):
            x, y, w, h = box
            size = w * h
            if size > max_size:
                max_size = size
                idx = i
        dstimg = yolov8.draw_detection(frame, boxes[idx], scores[idx], kpts[idx])
        cv2.imshow('Webcam', dstimg)
        

    # release
    model.release()
