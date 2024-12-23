from time import time
from rknn import RKNNPoolExecutor
from utils.camera import setup_camera
from func import func

import cv2
import os

MODEL_PATH = './models/yolo11n.rknn'
CAM_WIDTH = 640
CAM_HEIGHT = 640

WORKER_N = 4


if __name__ == '__main__':
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

    os.system('clear')
    while cap.isOpened():
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)
        frame, _ = pool.get()
        if frame is None:
            break
        cv2.imshow('rknn', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frames % 30 == 0:
            cur = time()
            if frames == 30:
                print(f"{30 / (cur - loop_time):.4f} FPS (warm-up)")
                init_time = cur
                loop_time = cur
                continue
            print(f"{30 / (time() - loop_time)} FPS")
            loop_time = time()
        if frames == 330:
            break

    print(f"\ntotal: {(frames - 30) / (time() - init_time):.4f} FPS (after warm-up)")

    cap.release()
    cv2.destroyAllWindows()
    pool.release()
