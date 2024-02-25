import os
import time
import sys
import shutil
from pathlib import Path
import pytest
import numpy as np
import torch
from cls.classification.utils.model_zoo import ClassificationModel


# class TestTorchScript:
#     MODEL_PATH = str(Path(__file__).parent / 'torchscript_dummy_model.pt')

#     def setup_method(self):
#         model = torch.nn.Linear(6, 4)
#         script = torch.jit.script(model)
#         torch.jit.save(script, self.MODEL_PATH)
    
#     def teardown_method(self):
#         if os.path.exists(self.MODEL_PATH):
#             os.remove(self.MODEL_PATH)

#     def test_process_batch(self):
#         model = ClassificationModel(self.MODEL_PATH, 1, inference_type='torchscript')
#         dummy_in = np.random.randn(1, 6).astype('float32')
#         out = model.process_batch(dummy_in)
#         assert out[0].shape == (1, 4)


class TestTriton:
    MODEL_DIR_PATH = str(Path(__file__).parent / 'model_repository' / 'triton_dummy_model')

    def setup_method(self):
        os.makedirs(os.path.join(self.MODEL_DIR_PATH, '1'), exist_ok=True)
        model = torch.nn.Linear(6, 4)
        trace_input = torch.randn((1, 6))
        model_path = os.path.join(self.MODEL_DIR_PATH, '1', 'model.onnx')
        torch.onnx.export(
            model,
            trace_input,
            model_path,
            verbose=True,
            export_params=True,     # store the trained parameter weights inside the model file
            opset_version=17,       # the ONNX version to export the model to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=["input"],    # the model's input names
            output_names=["output"],  # the model's output names
        )
        cfg_text = \
            "platform: \"onnxruntime_onnx\"\n"\
            "max_batch_size : 0\n"\
            "input [\n"\
            "{\n"\
            "    name: \"input\"\n"\
            f"    data_type: TYPE_FP32\n"\
            "    dims: [1, 6]\n"\
            "}\n"\
            "]\n"\
            "output [\n"\
            "{\n"\
            "    name: \"output\"\n"\
            f"    data_type: TYPE_FP32\n"\
            f"    dims: [1, 4]\n"\
            "}\n"\
            "]\n"

        cfg_path = os.path.join(self.MODEL_DIR_PATH, 'config.pbtxt')
        with open(cfg_path, 'w') as f:
            f.write(cfg_text)
        
        os.environ["TRITON_MODEL_REPOSITORY"] = os.path.dirname(self.MODEL_DIR_PATH)
        os.environ["TRITON_PORT"] = "8009"
        os.system(f"docker-compose -f {str(Path(__file__).parent / 'docker-compose.yaml')} up -d")
        time.sleep(2) # TODO: replace with normal checking

    
    def teardown_method(self):
        os.system(f"docker-compose -f {str(Path(__file__).parent / 'docker-compose.yaml')} down")
        shutil.rmtree(self.MODEL_DIR_PATH)

    def test_process_batch(self):
        model = ClassificationModel(
            os.path.basename(self.MODEL_DIR_PATH), 
            1, 
            inference_type='triton', 
            triton_url='0.0.0.0:8009')
        
        dummy_in = np.random.randn(1, 6).astype('float32')
        out = model.process_batch(dummy_in)
        assert out[0].shape == (1, 4)




if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
