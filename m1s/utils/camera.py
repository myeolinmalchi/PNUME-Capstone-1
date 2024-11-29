import cv2

def setup_camera(width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not capture.isOpened():
        raise Exception('카메라가 준비되지 않았습니다.')

    return capture
