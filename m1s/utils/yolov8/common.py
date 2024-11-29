import cv2

from typing import Literal, Tuple

from cv2.typing import MatLike


def draw_rectangle(image: MatLike, box, score):
    top, left, right, bottom = [int(_b) for _b in box]
    print("(%d %d %d %d) %.3f" % (top, left, right, bottom, score))
    cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
    cv2.putText(image, '{1:.2f}'.format(score),
                (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_rectangles(image: MatLike, boxes, scores):
    for box, score in zip(boxes, scores):
        draw_rectangle(image, box, score)


def get_largest_box(boxes) -> int:
    _idx = -1
    max_size = 0
    for idx, box in enumerate(boxes):
        top, left, right, bottom = [int(_b) for _b in box]
        size = (top - bottom) * (right - left)
        if size > max_size:
            _idx = idx

    return _idx


def draw_largest_box(image: MatLike, boxes, scores):
    idx = get_largest_box(boxes)
    box, score = boxes[idx], scores[idx]
    draw_rectangle(image, box, score)


from utils.wrappers import ModelWrapper 

def setup_model(args) -> Tuple[ModelWrapper, Literal['rknn', 'pytorch', 'onnx']]:
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from utils.wrappers import TorchWrapper
        model = TorchWrapper(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from utils.wrappers import RKNNWrapper
        model = RKNNWrapper(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from utils.wrappers import ONNXWrapper
        model = ONNXWrapper(args.model_path)

    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform
