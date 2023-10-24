import torch.optim as optim
from abc import ABC, abstractmethod
import lightning as L

class BaseModule(L.LightningModule, ABC):
    def __init__(self, model, lr=1e-3):
        super(BaseModule, self).__init__()
        self.model = model
        self.lr = lr

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
        return optim.AdamW(self.model.parameters(), lr=self.lr)

    