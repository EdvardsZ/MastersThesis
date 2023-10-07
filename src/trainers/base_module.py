import pytorch_lightning as pl
import torch.optim as optim
from abc import ABC, abstractmethod

class BaseModule(pl.LightningModule, ABC):
    def __init__(self, model):
        super(BaseModule, self).__init__()
        self.model = model

    @abstractmethod
    def forward(self, x, x_cond, y):
        pass
    @abstractmethod
    def step(self, batch, batch_idx, mode='train'):
        pass
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=1e-3)

    