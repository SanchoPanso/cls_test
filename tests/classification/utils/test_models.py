import os
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from cls.classification.utils.model_zoo import ClassificationModel


class TestTorchScript:
    MODEL_PATH = str(Path(__file__).parent / 'torchscript_dummy_model.pt')

    def setup_method(self):
        model = torch.nn.Linear(6, 4)
        script = torch.jit.script(model)
        torch.jit.save(script, self.MODEL_PATH)
    
    def teardown_method(self):
        if os.path.exists(self.MODEL_PATH):
            os.remove(self.MODEL_PATH)

    def test_process_batch(self):
        model = ClassificationModel(self.MODEL_PATH, 1, inference_type='torchscript')
        dummy_in = np.random.randn(1, 6).astype('float32')
        out = model.process_batch(dummy_in)
        assert out[0].shape == (1, 4)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
