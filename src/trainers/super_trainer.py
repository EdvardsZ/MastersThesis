from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L


class SuperTrainer(L.Trainer):
    def __init__(self, model_name: str, max_epochs: int, devices = [5]):
        self.model_name  = model_name

        self._epochs = max_epochs
        logger = TensorBoardLogger('lightning_logs/', name=self.model_name)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename= self.model_name + '_{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        super().__init__(accelerator='gpu', devices=devices, max_epochs = max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback], logger=logger)

    def fit(self, model, train_dataloader, val_dataloader):
        super().fit(model, train_dataloader, val_dataloader)

    def save_model_checkpoint(self):
        super().save_checkpoint('checkpoints/' + self.model_name + '.ckpt')
    