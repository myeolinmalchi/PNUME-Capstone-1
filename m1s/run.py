import argparse

from utils.yolov8.rknn import post_process
from utils.yolov8.common import draw_largest_box, setup_model
from utils.camera import setup_camera
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='./models/yolov8n-face.rknn', help='model path, could be .pt, .onnx or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    args = parser.parse_args()

    model, platform = setup_model(args)

    capture = setup_camera(640, 480)

    while capture.isOpened():
        status, frame = capture.read()
        
        if not status:
            break

        outputs = model.run([frame])
        boxes, classes, scores = post_process(outputs)

        if boxes is not None:
            draw_largest_box(frame, boxes, scores)

        cv2.imshow('Webcam', frame)
        

    # release
    model.release()
