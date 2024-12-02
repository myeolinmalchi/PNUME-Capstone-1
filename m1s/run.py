from utils.yolov8 import YOLOv8
from utils.wrappers import ModelWrapper
from utils.camera import setup_camera

import serial
import cv2

MODEL_PATH = './models/yolov8n-face.rknn'
CAM_WIDTH = 640
CAM_HEIGHT = 640

SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUDRATE = 9600


if __name__ == '__main__':
    model, _ = ModelWrapper.setup(MODEL_PATH)
    capture = setup_camera(CAM_WIDTH, CAM_HEIGHT)
    arduino = serial.Serial(
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUDRATE
    )
    yolov8 = YOLOv8(model)

    while cv2.waitKey(1) < 0:
        status, frame = capture.read()
        if not status:
            break

        result_img, pos = yolov8.detect_largest_face(frame)
        if pos is None:
            continue

        arduino.write(f"{pos[0]},{pos[1]}\n".encode())
        #cv2.imshow('Webcam', result_img)

    # release
    model.release()
