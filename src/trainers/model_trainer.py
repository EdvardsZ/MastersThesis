
from .base_module import BaseModule
from models import get_model

class ModelTrainer(BaseModule):
    def __init__(self, num_embeddings, embedding_dim):
        model_class = get_model('VQVAE')
        model = model_class(num_embeddings, embedding_dim)
        super(ModelTrainer, self).__init__(model)
        self.save_hyperparameters()
        
    def forward(self, x, x_cond, y):
        return self.model(x)
    
    def step(self, batch, batch_idx, mode = 'train'):
        x, x_cond, y = batch
        x_hat, quantized, latent, embedding_indices = self(x, x_cond, y)
        loss = self.model.loss(latent, quantized, x_hat, x)
        self.log_dict({f"{mode}_{key}": val.item() for key, val in loss.items()}, sync_dist=True, prog_bar=True)
        return loss['loss']
    
    def decode(self, z):
        return self.model.decode(z)
    
    def load_model_checkpoint(model_name):
        path = ('checkpoints/' + model_name + '.ckpt')
        return ModelTrainer.load_from_checkpoint(path)