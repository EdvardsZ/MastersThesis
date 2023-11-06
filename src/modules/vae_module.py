
from lightning_extensions import BaseModule
from models import get_model

class VAEModule(BaseModule):
    def __init__(self, model_params, image_shape, model_name):
        model_class = get_model(model_name)
        model = model_class(**model_params, image_shape = image_shape)
        super(VAEModule, self).__init__(model)
        self.save_hyperparameters()
        
    def forward(self, x, x_cond, y):
        return self.model(x, x_cond, y)
    
    def step(self, batch, batch_idx, mode = 'train'):
        x, x_cond, y = batch
        outputs = self(x, x_cond, y)
        loss = self.model.loss(batch, outputs)
        self.log_dict({f"{mode}_{key}": val.item() for key, val in loss.items()}, sync_dist=True, prog_bar=True)
        return loss['loss']
    
    def decode(self, z):
        return self.model.decode(z)
    
    @classmethod
    def load_model_checkpoint(self, model_name : str): 
        path = "checkpoints/" + model_name + ".ckpt"
        return VAEModule.load_from_checkpoint(path)