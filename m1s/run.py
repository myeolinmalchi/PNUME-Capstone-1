from time import time
from rknn import RKNNPoolExecutor
from utils.yolo11 import YOLO11
from utils.yolov8 import YOLOv8
from utils.wrappers import ModelWrapper
from utils.camera import setup_camera
from func import func

import serial
import cv2

MODEL_PATH = './models/yolov8n-face.rknn'
CAM_WIDTH = 640
CAM_HEIGHT = 640

SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUDRATE = 115200

WORKER_N = 4


'''
if __name__ == '__main__':
    model, _ = ModelWrapper.setup(MODEL_PATH)
    capture = setup_camera(CAM_WIDTH, CAM_HEIGHT)
    arduino = serial.Serial(
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUDRATE
    )
    yolov8 = YOLOv8(model)
    yolo11 = YOLO11(model)

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
'''

if __name__ == '__main__':
    rknn, _ = ModelWrapper.setup(MODEL_PATH)
    cap = setup_camera(CAM_WIDTH, CAM_HEIGHT)

    pool = RKNNPoolExecutor(MODEL_PATH, WORKER_N, func)

    if cap.isOpened():
        for i in range(WORKER_N + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                del pool
                exit(-1)
            pool.put(frame)

    frames, loop_time, init_time = 0, time(), time()

    while cap.isOpened():
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        frame, _ = pool.get()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('rknn', frame)

        if frames % 30 == 0:
            print(f"{30 / (time() - loop_time)} FPS")
            loop_time = time()

    print(f"\ntotal: {frames / (time() - init_time)} FPS")

    cap.release()
    cv2.destroyAllWindows()
    pool.release()
