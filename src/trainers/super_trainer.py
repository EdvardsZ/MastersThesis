from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


class SuperTrainer(pl.Trainer):
    def __init__(self, logger_name: str, max_epochs = int, devices = [5]):
        self.logger_name  = logger_name
        self._epochs = max_epochs
        logger = TensorBoardLogger('logs/', name=logger_name)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename= logger_name + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        super().__init__(accelerator='gpu', devices=devices, max_epochs = max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback], logger=logger)

    def fit(self, model, train_dataloader, val_dataloader):
        super().fit(model, train_dataloader, val_dataloader)

    def save_model_checkpoint(self):
        super().save_checkpoint('checkpoints/' + self.logger_name + '_' + self._epochs  + '.ckpt')
    