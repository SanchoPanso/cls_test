import os
import sys
import logging
import wandb
import torch
import json
import numpy as np
import zipfile
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer as BaseTrainer

from cls.classification.engine.model import EfficientLightning
from cls.classification.engine.wrapper import TrainWrapper
from cls.classification.utils.logger import get_logger

LOGGER = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    
    def __init__(self, wrapper: TrainWrapper):
        callbacks = wrapper.get_callbacks()
        
        loggers = wrapper.get_loggers()
        self.wandb_logger, self.csv_logger = loggers
        
        self.wrapper = wrapper
        
        super().__init__(
            accelerator="gpu",
            devices=[wrapper.gpu],
            max_epochs=wrapper.epochs,
            precision=16,
            log_every_n_steps=1,
            logger=loggers,
            callbacks=callbacks,
        )
    
    def fit(self, model: EfficientLightning):
        try:
            self._create_train_examples(self.wrapper)
            super().fit(model, self.wrapper.train_loader, self.wrapper.val_loader)
        
        except Exception as e:
            LOGGER.error(e)
            
        finally:
            paths = self.wrapper.get_model_paths()
            if len(paths) == 0:
                return
            
            torchscript_path = paths[0]
            name = os.path.splitext(os.path.basename(torchscript_path))[0]
            onnx_dir = os.path.join(self.wrapper.save_dir, self.wrapper.cat, self.wrapper.experiment_name, "onnx", name)
            converter = ClsOnnxConverter()
            converter(torchscript_path, onnx_dir, fp16=True, to_zip_result=True)

            artifact = wandb.Artifact(name=self.wandb_logger._checkpoint_name, type="model")
            artifact.add_file(torchscript_path, name="model.pt")
            artifact.add_file(onnx_dir + '.zip', name="model.zip")
            aliases = ["latest", "best"]
            self.wandb_logger.experiment.log_artifact(artifact, aliases=aliases)
    
        
    def _create_train_examples(self, wrapper: TrainWrapper, num_of_batches=1):
        train_batches_dir = os.path.join(wrapper.save_dir, wrapper.cat, wrapper.experiment_name, 'train_batches')
        os.makedirs(train_batches_dir, exist_ok=True)
        
        for i, batch in enumerate(wrapper.train_loader):
            if i >= num_of_batches:
                break
            
            imgs, labels = batch
            batch_size = imgs.shape[0]
            height = int(np.ceil(np.sqrt(batch_size)))

            plt.figure(figsize=(45, 15))
            for j in range(batch_size):
                img = imgs[j].permute(1, 2, 0).cpu().float().numpy()
                label = labels[j].cpu().numpy().argmax()
                name = wrapper.num2label[str(label)]
                
                plt.subplot(height, height, j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.title(name)
                plt.imshow(img)
                
            plt.savefig(os.path.join(train_batches_dir, f'train_batch_{i}.jpg'))                 
            

class ClsOnnxConverter:
    def __init__(self) -> None:
        pass
        
    def __call__(
        self, 
        src_path: str, 
        dst_path: str, 
        fp16: bool, 
        size: tuple = (480, 640), 
        device: str = 'cuda',
        to_zip_result: bool = False) -> None:
        
        os.makedirs(os.path.join(dst_path, '1'), exist_ok=True)
        
        model_path = os.path.join(dst_path, '1', 'model.onnx')
        extra_path = os.path.join(dst_path, '1', 'meta.json')
        
        model, extra = self.get_model(src_path)
        num_of_classes = len(extra['num2label.txt'])
        print(extra)
        
        self.save_model(model, extra, model_path, extra_path, fp16, size)
        self.save_cfg(os.path.join(dst_path, 'config.pbtxt'), fp16, size, num_of_classes)

        if to_zip_result:
            self.zip_result(dst_path)
    
    def save_model(self, model: torch.nn.Module, extra: dict, model_path: str, extra_path: str, fp16: bool, size: tuple):
        trace_input = self.get_sample(size)
        
        if fp16:
            trace_input = trace_input.half()
            model = model.half()
        
        self.save_onnx(model, trace_input, model_path)
        self.save_extra(extra, extra_path)
    
    def get_model(self, model_path: str, device: str = 'cuda'):
        extra = {"num2label.txt": ""}
        model = torch.jit.load(model_path, device, _extra_files=extra)
        model = model.eval()
        extra["num2label.txt"] = json.loads(extra["num2label.txt"])
        return model, extra
    
    def save_onnx(self, model: torch.nn.Module, trace_input: torch.Tensor, path: str):
        torch.onnx.export(
            model,
            trace_input,
            path,
            verbose=True,
            export_params=True,     # store the trained parameter weights inside the model file
            opset_version=10,       # the ONNX version to export the model to
            do_constant_folding=True,   # whether to execute constant folding for optimization
            input_names=["input.1"],    # the model's input names
            output_names=["output.1"],  # the model's output names
            dynamic_axes={'input.1' : {0 : 'batch_size'},   # variable length axes
                        'output.1' : {0 : 'batch_size'}}
        )
    
    def save_extra(self, extra: dict, path: str):
        with open(path, 'w') as f:
            json.dump(extra, f)
    
    def zip_result(self, path: str):
        with zipfile.ZipFile(f"{path}.zip", mode="w") as archive:
            directory = Path(path)
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))


    def get_sample(self, size: Tuple[int, int] = (640, 480), device: str = 'cuda'):
        trace_input = torch.randn(1, 3, size[0], size[1]).to(device)
        return trace_input
    
    def save_cfg(self, path: str, fp16: bool, size: tuple = (480, 640), num_of_classes: int = 0):
        data_type = "TYPE_FP16" if fp16 else "TYPE_FP32"
        cfg_text = \
            "backend: \"onnxruntime\"\n"\
            "max_batch_size : 256\n"\
            "input [\n"\
            "{\n"\
            "    name: \"input.1\"\n"\
            f"    data_type: {data_type}\n"\
            "    dims: [3, 480, 640]\n"\
            "}\n"\
            "]\n"\
            "output [\n"\
            "{\n"\
            "    name: \"output.1\"\n"\
            f"    data_type: {data_type}\n"\
            f"    dims: [{num_of_classes}]\n"\
            "}\n"\
            "]\n"\
            "model_warmup [\n"\
            "    {\n"\
            "        name : \"images\",\n"\
            "        batch_size: 50,\n"\
            "        inputs {\n"\
            "            key: \"input.1\"\n"\
            "            value: {\n"\
            f"                data_type: {data_type}\n"\
            f"                dims: [ 3, {size[0]}, {size[1]} ]\n"\
            "                random_data: true\n"\
            "            }\n"\
            "        }\n"\
            "    }\n"\
            "]\n"

        
        with open(path, 'w') as f:
            f.write(cfg_text)
            
