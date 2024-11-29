import os
import numpy as np
import onnxruntime as rt
import torch

from abc import abstractmethod, ABC
from typing import Any, Generic, TypeVar, Union, Tuple, List
from rknn.api import RKNN


torch.backends.quantized.engine = 'qnnpack'

I = TypeVar("I")
O = TypeVar("O")

class ModelWrapper(ABC, Generic[I, O]):
    @abstractmethod
    def __init__(self, model_path: str, target = None, device_id = None):  ...

    @abstractmethod
    def run(self, inputs: Union[I, List[I], Tuple[I, ...]]) -> O: ...

    @abstractmethod
    def release(self): ...



class RKNNWrapper(ModelWrapper[Any, Any]):
    """RKNN 모델 랩퍼"""

    def __init__(self, model_path, target, device_id) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None



class TorchWrapper(ModelWrapper[Any, Any]):
    """pytorch 모델 랩퍼"""
    @staticmethod
    def flatten_list(in_list):
        flatten = lambda x: [subitem for item in x for subitem in flatten(item)] if type(x) is list else [x]
        return flatten(in_list)

    def __init__(self, model_path, qnnpack=False) -> None:
        if qnnpack is True:
            torch.backends.quantized.engine = 'qnnpack'

        #! Backends must be set before load model.
        self.pt_model = torch.jit.load(model_path)
        self.pt_model.eval()

    def run(self, inputs):
        if self.pt_model is None:
            print("ERROR: pt_model has been released")
            return []

        assert isinstance(inputs, list), "input_datas should be a list, like [np.ndarray, np.ndarray]"

        input_datas_torch_type = []
        for _data in inputs:
            input_datas_torch_type.append(torch.tensor(_data))

        for i,val in enumerate(input_datas_torch_type):
            if val.dtype == torch.float64:
                input_datas_torch_type[i] = input_datas_torch_type[i].float()

        result = self.pt_model(*input_datas_torch_type)

        if isinstance(result, tuple):
            result = list(result)
        if not isinstance(result, list):
            result = [result]

        result = self.flatten_list(result)

        for i in range(len(result)):
            result[i] = torch.dequantize(result[i])

        for i in range(len(result)):
            # TODO support quantized_output
            result[i] = result[i].cpu().detach().numpy()

        return result

    def release(self):
        del self.pt_model
        self.pt_model = None


type_map = {
    'tensor(int32)' : np.int32,
    'tensor(int64)' : np.int64,
    'tensor(float32)' : np.float32,
    'tensor(float64)' : np.float64,
    'tensor(float)' : np.float32,
}
if getattr(np, 'bool', False):
    type_map['tensor(bool)'] = np.bool_
else:
    type_map['tensor(bool)'] = bool



class ONNXWrapper(ModelWrapper[Any, Any]):

    @staticmethod
    def ignore_dim_with_zero(_shape, _shape_target):
        _shape = list(_shape)
        _shape_target = list(_shape_target)
        for _ in range(_shape.count(1)):
            _shape.remove(1)
        for _ in range(_shape_target.count(1)):
            _shape_target.remove(1)
        if _shape == _shape_target:
            return True
        else:
            return False

    @staticmethod
    def reset_onnx_shape(onnx_model_path, output_path, input_shapes):
        if isinstance(input_shapes[0], int):
            command = "python -m onnxsim {} {} --input-shape {}".format(onnx_model_path, output_path, ','.join([str(v) for v in input_shapes]))
        else:
            if len(input_shapes)!= 1:
                print("RESET ONNX SHAPE with more than one input, try to match input name")
                sess = rt.InferenceSession(onnx_model_path)
                input_names = [input.name for input in sess.get_inputs()]
                command = "python -m onnxsim {} {} --input-shape ".format(onnx_model_path, output_path)
                for i, input_name in enumerate(input_names):
                    command += "{}:{} ".format(input_name, ','.join([str(v) for v in input_shapes[i]]))
            else:
                command = "python -m onnxsim {} {} --input-shape {}".format(onnx_model_path, output_path, ','.join([str(v) for v in input_shapes[0]]))
        
        print(command)
        os.system(command)
        return output_path
    
    def __init__(self, model_path) -> None:
        # sess_options=
        sp_options = rt.SessionOptions()
        sp_options.log_severity_level = 3
        # [1 for info, 2 for warning, 3 for error, 4 for fatal] 
        self.sess = rt.InferenceSession(model_path, sess_options=sp_options, providers=['CPUExecutionProvider'])
        self.model_path = model_path

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.sess is None:
            print("ERROR: sess has been released")
            return []

        if isinstance(inputs, list):
            pass
        elif isinstance(inputs, tuple):
            inputs = list(inputs)
        else:
            inputs = [inputs]

        if len(inputs) < len(self.sess.get_inputs()):
            assert False,'inputs_datas number not match onnx model{} input'.format(self.model_path)
        elif len(inputs) > len(self.sess.get_inputs()):
            print('WARNING: input datas number large than onnx input node')


        input_dict = {}
        for i, _input in enumerate(self.sess.get_inputs()):
            # convert type
            if _input.type in type_map and \
                type_map[_input.type] != inputs[i].dtype:
                print('WARNING: force data-{} from {} to {}'.format(i, inputs[i].dtype, type_map[_input.type]))
                inputs[i] = inputs[i].astype(type_map[_input.type])
            
            # reshape if need
            if _input.shape != list(inputs[i].shape):
                if self.ignore_dim_with_zero(inputs[i].shape,_input.shape):
                    inputs[i] = inputs[i].reshape(_input.shape)
                    print("WARNING: reshape inputdata-{}: from {} to {}".format(i, inputs[i].shape, _input.shape))
                else:
                    assert False, 'input shape{} not match real data shape{}'.format(_input.shape, inputs[i].shape)
            input_dict[_input.name] = inputs[i]

        output_list = []
        for i in range(len(self.sess.get_outputs())):
            output_list.append(self.sess.get_outputs()[i].name)

        #forward model
        res = self.sess.run(output_list, input_dict)
        return res

    def release(self):
        del self.sess
        self.sess = None
