import os
import torch
import numpy as np
import cv2
from easydict import EasyDict
from abc import ABC
import numpy as np
from numpy import ndarray
import onnxruntime as ort
import tritonclient.http as httpclient
from typing import Tuple, List, Dict, Sequence, Any
from typing import Any, List


class Inferencer(ABC):
    """Implementor of inference logic"""

    def __call__(self, 
                 input_dict: Dict[str, np.ndarray],
                 model_outs: Sequence[str] = None,
                 *args, 
                 **kwargs) -> List[np.ndarray]:
        """Perform inference

        Args:
            input_dict (Dict[str, np.ndarray]): dictionary of input arrays
            model_outs (Sequence[str]): sequence of model output names

        Returns:
            List[np.ndarray]: list of output arrays
        """
        pass


class TritonInferencer(Inferencer):
    """Concrete implementor of inference for triton server client. 
    This class implements the specific logic for performing inference using Triton Server.

    Attributes:
        model_name (str): The name of the model to be used for inference.
        client (httpclient.InferenceServerClient): The Triton Server client instance.
    """

    def __init__(self, model_path: str, triton_url: str = 'localhost:8000', *args, **kwargs) -> None:
        """
        Args:
            model_name (str): The name of the model to be used for inference.
            url (str): The URL of the Triton Server. # TODO: default
        """
        self.model_path = model_path
        self.client = httpclient.InferenceServerClient(url=triton_url)
        
    def __call__(self, 
                 input_dict: Dict[str, np.ndarray] | np.ndarray,
                 model_outs: Sequence[str] = None,
                 *args, 
                 **kwargs) -> List[np.ndarray]:
        """
        Perform inference using Triton Server.

        Args:
            input_dict (Dict[str, np.ndarray]): A dictionary of input arrays.
            model_outs (Sequence[str]): A sequence of model output names.

        Returns:
            Dict[str, np.ndarray]: A dictionary of output arrays with keys as output names.
        """
        
        # Setting up input and output
        inputs = []
        for name in input_dict:
            datatype = "FP32" if input_dict[name].dtype == np.float32 else "FP16"
            inp = httpclient.InferInput(name, input_dict[name].shape, datatype=datatype)
            inp.set_data_from_numpy(input_dict[name], binary_data=True)
            inputs.append(inp)

        outputs = []
        for name in model_outs:
            out = httpclient.InferRequestedOutput(name, binary_data=True)
            outputs.append(out)
            
        # Querying the server
        results = self.client.infer(
            model_name=self.model_path, 
            inputs=inputs, 
            outputs=outputs
        )

        inference_outputs = [results.as_numpy(name) for name in model_outs]
        return inference_outputs


class ONNXInferencer(Inferencer):
    """Concrete implementor of inference for onnxruntime session. 
    This class implements the specific logic for performing inference using ONNX Runtime.

    Attributes:
        model_path (str): The path to the ONNX model file.
        providers (Sequence[str]): A sequence of providers for ONNX Runtime.
    """
    # TODO: add provider examples

    def __init__(self, model_path: str, providers: Sequence[str] = None, *args, **kwargs) -> None:
        """
        Args:
            model_path (str): The path to the ONNX model file.
            providers (Sequence[str]): A sequence of providers for ONNX Runtime (e.g., ...).
        """
        super().__init__()
        self.ort = ort.InferenceSession(model_path, providers=providers)
    
    def __call__(self, input_dict: Dict[str, np.ndarray] | np.ndarray, check_dims=True, *args, **kwargs) -> List[np.ndarray]:
        """
        Perform inference using ONNX Runtime.

        Args:
            input_dict (Dict[str, np.ndarray]): A dictionary of input arrays.
            check_dims (bool): Whether to check and adjust input dimensions. Default is True.

        Returns:
            Dict[str, np.ndarray]: A dictionary of output arrays with keys as output names.
        """

        if check_dims:
            input_dict = self.check_dims(input_dict)
        outputs = self.ort.run(None, input_dict)
        return outputs
    
    def check_dims(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key in input_dict:
            if len(input_dict[key].shape) == 3:
                input_dict[key] = np.expand_dims(input_dict[key], 0)
        return input_dict


class TorchscriptInferencer(Inferencer):
    
    def __init__(self, model_path: str, device: str = None, *args, **kwargs) -> None:
        super().__init__()
        self.extra_files: dict = {"num2label.txt": ""}
        self.device: str = device
        self.model: torch.nn.Module = torch.jit.load(model_path, device, _extra_files=self.extra_files)
        self.model.eval()
    
    def __call__(self, input_dict: Dict[str, np.ndarray], *args, **kwargs) -> List[np.ndarray]:
        if len(input_dict) != 1:
            raise ValueError
        
        input_arr = list(input_dict.values())[0]
        input_tensor = torch.from_numpy(input_arr).to(self.device)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        output_arr = output_tensor.cpu().numpy()
        return [output_arr]
    

class Model(ABC):
    """Abstraction of a ML model.

    This class serves as an abstraction for machine learning models, utilizing the Bridge pattern
    with the `Inferencer` class as the implementor.

    Attributes:
        inferencer (Inferencer): The implementor for inference logic.
        batch_size (int): The batch size used during inference (if dynamic is True - it is a max batch size).
        dynamic (bool): Flag indicating whether the model supports dynamic batch sizes.

    """
    
    inferencer: Inferencer
    batch_size: int
    dynamic: bool

    def __init__(
            self, 
            model_path: str, 
            batch_size: int = 1, 
            dynamic: bool = False, 
            inference_type: str = 'triton', 
            *args, 
            **kwargs,
        ):

        # TODO: redo docs
        """
        Args:
            inferencer (Inferencer): The implementor for inference logic.
            batch_size (int): The batch size used during inference. Default is 1.
            dynamic (bool): Flag indicating whether the model supports dynamic batch sizes. Default is False.
        """

        self.inferencer = self.get_inferencer(model_path, inference_type, *args, **kwargs)
        self.batch_size = batch_size
        self.dynamic = dynamic


    def __call__(self, input_arrays: List[np.ndarray], *args, **kwargs) -> Any:
        """
        Performs inference using the specified input arrays.

        Args:
            input_arrays (List[np.ndarray]): List of input arrays for inference.

        Returns:
            Any: The result of the inference.
        """
        pass

    def get_inferencer(
            self, 
            model_path: str, 
            inference_type: str,
            *args, 
            **kwargs) -> Inferencer:
        
        assert inference_type in ['triton', 'torchscript']
        if inference_type == 'triton':
            inferencer = TritonInferencer(model_path, *args, **kwargs)
        else:
            inferencer = TorchscriptInferencer(model_path, *args, **kwargs)
        
        return inferencer

    def build_batches(self, preprocessed_arrays: List[np.ndarray]) -> List[np.ndarray]:

        batches = []
        if len(preprocessed_arrays) == 0:
            return batches
        
        batch_size = self.batch_size
        dynamic = self.dynamic
    
        num_of_batches = int(np.ceil(len(preprocessed_arrays) / batch_size))

        for i in range(num_of_batches):
            start_idx = batch_size * i
            stop_idx = min(batch_size * (i + 1), len(preprocessed_arrays))
            batch_arrays = preprocessed_arrays[start_idx: stop_idx]
            batch = np.array(batch_arrays)
            
            num_to_fill = max(0, batch_size - len(batch_arrays))
            
            if num_to_fill != 0 and not dynamic:
                zero_arrays = np.zeros((num_to_fill, *batch_arrays[0]), dtype=batch_arrays[0].dtype)
                batch = np.concatenate([batch, zero_arrays], axis=0)

            batches.append(batch)
        
        return batches


class ClassificationModel(Model):
    def __init__(
            self, 
            model_path: str, 
            batch_size: int = 1, 
            dynamic: bool = False, 
            inference_type: str = 'triton',
            classes: List[str] = None, 
            fp16: bool = True,
            *args, 
            **kwargs,
        ):

        super().__init__(model_path, batch_size, dynamic, inference_type, *args, **kwargs)
        self.classes = classes or []
        self.fp16 = fp16


    def process_batch(self, batch: np.ndarray) -> Sequence[np.ndarray]:
        batch = batch.astype('float16') if self.fp16 else batch.astype('float32')
        output = self.inferencer({'input.1': batch}, ['output.1'], datatype="FP16")
        return output
    
    def get_class_name(self, idx: int):
        if idx >= len(self.classes):
            return str(idx)
        return self.classes[str(idx)]
    

class ModelZoo:
    __shared_models = {}
    
    def __init__(self, cfg: EasyDict):
        self.cfg = cfg

    def get_cls_model(self, model_type: str, inference_type: str = 'torchscript') -> Model:
        
        # identifier = f'{backend}://{model_type}'
        # if identifier not in self.__shared_models:
        #     self.__shared_models[identifier] = None
        
        # model = self.__shared_models[identifier]

        triton_url = os.getenv('TRITON_URL')
        if inference_type == 'triton':
            model_path = model_type
            fp16 = True
        elif inference_type == 'torchscript':
            model_path = self.cfg['MODELS'][model_type]
            fp16 = False
        else:
            raise ValueError
        
        classes = self.cfg['CLASSES'][model_type]
        model = ClassificationModel(model_path, 
                                    classes=classes,
                                    fp16=fp16,
                                    inference_type=inference_type, 
                                    triton_url=triton_url)
        return model
