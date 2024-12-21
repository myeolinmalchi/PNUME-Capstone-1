'''
Convert ONNX to RKNN
https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolo11/python/convert.py
'''


from rknn.api import RKNN
import argparse

DATASET_PATH = './datasets/COCO/coco_subset_20.txt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_path", action="store", default="./models/yolo11n.onnx")
    parser.add_argument("-p", dest="platform", action="store", default="rk3566")
    parser.add_argument("-o", dest="output_path", action="store", default="./models/yolo11n.rknn")
    parser.add_argument("-d", dest="dtype", action="store", default="int8")
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=args.platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    do_quant = args.dtype in ['int8', 'uint8']
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
