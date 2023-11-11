import os
import sys
import logging
import wandb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer as BaseTrainer

sys.path.append(str(Path(__file__).parent.parent))
from engine.model import EfficientLightning
from engine.wrapper import TrainWrapper
from utils.logger import get_logger

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
            
            artifact = wandb.Artifact(name=self.wandb_logger._checkpoint_name, type="model")
            artifact.add_file(paths[0], name="model.pt")
            aliases = ["latest", "best"]
            self.wandb_logger.experiment.log_artifact(artifact, aliases=aliases)
        
    def _create_train_examples(self, wrapper: TrainWrapper, num_of_batches=3):
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

