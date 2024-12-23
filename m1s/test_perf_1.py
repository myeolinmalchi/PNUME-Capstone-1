from time import time
from utils.yolo11 import YOLO11
from utils.wrappers import ModelWrapper
from utils.camera import setup_camera
import cv2
import os

MODEL_PATH = './models/yolo11n.rknn'
CAM_WIDTH = 640
CAM_HEIGHT = 640

if __name__ == '__main__':
    model, _ = ModelWrapper.setup(MODEL_PATH)
    cap = setup_camera(CAM_WIDTH, CAM_HEIGHT)
    yolo11 = YOLO11(model)
    frames, init_time, loop_time = 0, time(), time()

    os.system('clear')
    while cv2.waitKey(1) < 0:
        frames += 1
        status, frame = cap.read()
        if not status:
            break
        result_img, pos = yolo11.detect_largest_object(frame)
        cv2.imshow('Webcam', result_img)
        if frames % 30 == 0:
            cur = time()
            if frames == 30:
                print(f"{30 / (cur - loop_time):.4f} FPS (warm-up)")
                init_time = cur
                loop_time = cur
                continue
            
            print(f"{30 / (cur - loop_time):.4f} FPS")
            loop_time = cur
        if frames == 330:
            break

    print(f"\ntotal: {(frames - 30) / (time() - init_time):.4f} FPS (after warm-up)")

    cap.release()
    cv2.destroyAllWindows()
    model.release()
