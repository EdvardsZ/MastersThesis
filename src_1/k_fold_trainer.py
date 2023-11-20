from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import lightning as L
import wandb
from pl_crossvalidate import KFoldDataModule
from torch.utils.data import DataLoader, Subset
import os.path as osp


class KFoldTrainer(L.Trainer):
    def __init__(self, project_name: str, model_name: str, max_epochs: int, num_folds: int = 5, devices = [5], monitor = "val_loss", **kwargs ):
        self.model_name  = model_name

        self.project_name = project_name
        self._epochs = max_epochs
        self.num_folds = num_folds
        self.shuffle = False # FOR NOW
        self.stratified = False # FOR NOW


        logger = TensorBoardLogger(save_dir='lightning_logs/', name=self.model_name)
        self.fold = 0
        self.wandb = WandbLogger(project = project_name, name=self.get_fold_model_name(), log_model="all", group = self.model_name)

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath='checkpoints/',
            filename= self.model_name + '_{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        super().__init__(accelerator='gpu', devices=devices, max_epochs = max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback], logger=[logger, self.wandb], **kwargs)

    def _construct_kfold_datamodule(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: DataLoader,
    ) -> KFoldDataModule:
        return KFoldDataModule(
            self.num_folds,
            self.shuffle,
            self.stratified,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=None,
        )

    def crossvalidate(self, model, train_dataloader, val_dataloader):
        print("Starting crossvalidation")

        data_module = self._construct_kfold_datamodule(train_dataloader, val_dataloader)


        # checkpoint to restore from
        # this is a bit hacky because the model needs to be saved before the fit method
        self.strategy._lightning_module = model
        path = osp.join(self.log_dir, "kfold_initial_weights.ckpt")
        self.save_checkpoint(path)
        self.strategy._lightning_module = None


        results = []
        for i in range(self.num_folds):
            data_module.fold_index = i
            self.fold = i
            self.logger = WandbLogger(project = self.project_name, name=self.get_fold_model_name(), log_model="all", group = self.model_name)
            self.fit(model, data_module, ckpt_path=path)
            self.save_model_checkpoint(self.get_fold_model_name())
            res = self.test(model=model, datamodule=datamodule, verbose=False)
            results.append(res)
            self.logger.finalize("success")
            self.wandb.finalize("success")
            wandb.finish(quiet=True)

        return results


    def save_model_checkpoint(self, model_name = None):
        model_name = self.get_fold_model_name() if model_name is None else model_name
        super().save_checkpoint('checkpoints/' + model_name + '.ckpt')


    def get_fold_model_name(self):
        return self.model_name + "_fold_" + str(self.fold)
        