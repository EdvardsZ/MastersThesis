import pytorch_lightning as pl
import torch.optim as optim
from abc import ABC, abstractmethod

class BaseTrainer(pl.LightningModule, ABC):
    def __init__(self, model):
        super(BaseTrainer, self).__init__()
        self.model = model


    # override if needed  
    def forward(self, x, x_cond, y):
        return self.model(x, x_cond, y)
    
    # override if needed
    def step(self, batch, batch_idx, mode='train'):
        x, x_cond, y = batch
        x_hat, *other_outputs = self(x, x_cond, y)
        loss = self.model.loss(x, x_hat, *other_outputs)
        self.log_dict({f"{mode}_{key}": val.item() for key, val in loss.items()}, sync_dist=True, prog_bar=True)
        return loss['loss']
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)

    