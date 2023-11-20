import torch.optim as optim
from abc import ABC, abstractmethod
import lightning as L
from vae import VAE

class BaseModule(L.LightningModule, ABC):
    def __init__(self, lr=1e-3):
        super(BaseModule, self).__init__()
        self.save_hyperparameters() # this means we can load the model from a checkpoint without having to specify the model parameters again

        self.model = VAE(784, 512, 256, 2)
        self.lr = lr

    def forward(self, x, y):
        return self.model(x, y)
    
    def step(self, batch, batch_idx, mode = 'train'):
        x, y = batch
        outputs = self(x,y)
        loss = self.model.loss(batch, outputs)
        self.log_dict({f"{mode}_{key}": val.item() for key, val in loss.items()}, sync_dist=True, prog_bar=True)
        return loss['loss']
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
    
    @classmethod
    def load_model_checkpoint(self, model_name : str): 
        path = "checkpoints/" + model_name + ".ckpt"
        return BaseModule.load_from_checkpoint(path)