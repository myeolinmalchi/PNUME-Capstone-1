import torch
from abc import abstractmethod, ABC
from typing import Any, Generic, TypeVar, Union, Tuple, List
from rknnlite.api import RKNNLite


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

    @staticmethod
    def setup(model_path: str):
        if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
            platform = 'pytorch'
            from utils.wrappers import TorchWrapper
            model = TorchWrapper(model_path)
        elif model_path.endswith('.rknn'):
            platform = 'rknn'
            from utils.wrappers import RKNNWrapper
            model = RKNNWrapper(model_path)

        else:
            assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
        print('Model-{} is {} model, starting val'.format(model_path, platform))
        return model, platform



class RKNNWrapper(ModelWrapper[Any, Any]):

    def __init__(self, model_path) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        ret = rknn.init_runtime()
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
