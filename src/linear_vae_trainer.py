import pytorch_lightning as pl
from models import LinearVAE
import torch

class LinearVAETrainer(pl.LightningModule):
    def __init__(self):
        super(LinearVAETrainer, self).__init__()
        self.model = LinearVAE()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        train_loss = self.loss(x_hat, x, mu, log_var)
        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        val_loss = self.loss(x_hat, x, mu, log_var)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        return val_loss['loss']
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        test_loss = self.loss(x_hat, x, mu, log_var)
        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)
        return test_loss['loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    