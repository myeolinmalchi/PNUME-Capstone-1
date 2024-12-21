'''
Convert PyTorch to ONNX
https://github.com/airockchip/ultralytics_yolo11
'''

from ultralytics.cfg import IterableSimpleNamespace

DEFAULT_CFG = IterableSimpleNamespace(**{
    "keras": False, 
    "optimize": False, 
    "int8": False, 
    "dynamic": False, 
    "simplify": True, 
    "workspace": 4, 
    "nms": False, 
})


def export(cfg=DEFAULT_CFG):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_path", action="store", default="./models/yolo11n.pt")
    args = parser.parse_args()

    cfg.model = args.model_path
    cfg.format = "rknn"

    from ultralytics import YOLO
    model = YOLO(cfg.model)
    model.export(**vars(cfg))

if __name__ == '__main__':
    export()
