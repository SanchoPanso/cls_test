import os
import torch
import numpy as np
import cv2
from abc import ABC
import numpy as np
from numpy import ndarray
import onnxruntime as ort
import tritonclient.http as httpclient
from typing import Tuple, List, Dict, Sequence, Any
from typing import Any, List
AVAILABLE_BACKENDS = ['torchscript', 'triton']




class Inferencer(ABC):
    """Implementor of inference logic"""

    def __call__(self, 
                 input_dict: Dict[str, np.ndarray],
                 model_outs: Sequence[str],
                 *args, 
                 **kwargs) -> Dict[str, np.ndarray]:
        """Perform inference

        Args:
            input_dict (Dict[str, np.ndarray]): dictionary of input arrays
            model_outs (Sequence[str]): sequence of model output names

        Returns:
            Dict[str, np.ndarray]: dictionary of output arrays with keys - output names
        """
        pass


class TritonInferencer(Inferencer):
    """Concrete implementor of inference for triton server client. 
    This class implements the specific logic for performing inference using Triton Server.

    Attributes:
        model_name (str): The name of the model to be used for inference.
        client (httpclient.InferenceServerClient): The Triton Server client instance.
    """

    def __init__(self, model_path: str, triton_url: str = 'localhost:8000') -> None:
        """
        Args:
            model_name (str): The name of the model to be used for inference.
            url (str): The URL of the Triton Server. # TODO: default
        """
        self.model_path = model_path
        self.client = httpclient.InferenceServerClient(url=triton_url)
        
    def __call__(self, 
                 input_dict: Dict[str, np.ndarray],
                 model_outs: Sequence[str],
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
            inp = httpclient.InferInput(name, input_dict[name].shape, datatype="FP32")
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

    def __init__(self, model_path: str, providers: Sequence[str] = None) -> None:
        """
        Args:
            model_path (str): The path to the ONNX model file.
            providers (Sequence[str]): A sequence of providers for ONNX Runtime (e.g., ...).
        """
        super().__init__()
        self.ort = ort.InferenceSession(model_path, providers=providers)
    
    def __call__(self, input_dict: Dict[str, np.ndarray], check_dims=True, *args, **kwargs) -> Dict[str, np.ndarray]:
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
    
    def __init__(self, model_path: str, device: str = None) -> None:
        super().__init__()
        self.extra_files: dict = {"num2label.txt": ""}
        self.device: str = device
        self.model: torch.nn.Module = torch.jit.load(model_path, device, _extra_files=self.extra_files)
        self.model.eval()
    
    def __call__(self, input_dict: Dict[str, np.ndarray], *args, **kwargs) -> Dict[str, np.ndarray]:
        input_arr = list(input_dict.values())[0]
        input_tensor = torch.from_numpy(input_arr).to(self.device)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        output_arr = output_tensor.cpu().numpy()
        return (output_arr,)
    

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
    def process_batch(self, batch: np.ndarray) -> Sequence[np.ndarray]:
        output = self.inferencer({'input': batch}, ['output'])
        return output

class ModelZoo:
    __shared_models = {}
    
    def __init__(self):
        pass

    def get_model(self, path: str, backend: str = 'torch') -> 'Model':
        assert backend in AVAILABLE_BACKENDS        
        
        identifier = f'{backend}://{path}'
        if identifier not in self.__shared_models:
            self.__shared_models[identifier] = None #create_engine(url, echo=echo)
        
        model = self.__shared_models[identifier]
        return model


