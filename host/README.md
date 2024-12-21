## YOLO11 Format Conversion Tool

- Python >= 3.8
- refs
  - [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolo11#3-pretrained-model)
  - [airockchip/ultralytics_yolo11](https://github.com/airockchip/ultralytics_yolo11)

###  Usage

1. 의존성 설치
```bash
$ git clone https://github.com/myeolinmalchi/PNUME-Capstone-1.git
$ cd PNUME-Capstone-1/host
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

2. PyTorch -> ONNX
```bash
$ python convert_to_onnx.py -m <model_path>
    -m: your pytorch model path (default: ./models/yolo11n.pt)
```

3. ONNX -> RKNN
```bash
$ python convert_to_rknn.py -m <model_path> -p <platform> -o <output_path> -d <dtype>
    -m: onnx model path (default: ./models/yolo11n.onnx)
    -p: platform (rk356x/rk3588/rk3576)
    -o: rknn model output path (default: ./models/yolo11n.rknn)
    -d: dtype (int8/uint8/fp16)
```
