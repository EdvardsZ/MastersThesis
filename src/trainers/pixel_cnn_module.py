
from .base_module import BaseModule
from .vae_module import VAEModule
from models.autoregresive import SimplePixelCNN
import torch.nn as nn
from torch.autograd import Variable

class PixelCNNModule(BaseModule):
    def __init__(self, vae_name):

        # vae does not require to be trained and saved
        vae = VAEModule.load_model_checkpoint(vae_name)
        vae.freeze()
        vae.eval()

        model = SimplePixelCNN()
        super(PixelCNNModule, self).__init__(model)
        self.vae = vae
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, batch_idx, mode = 'train'):
        x, x_cond, y = batch
        x_hat, quantized, latent, embedding_indices= self.vae(x, x_cond, y)
        
        # reshape back embedding indices and detach from graph
        embedding_indices = embedding_indices.reshape(shape=(latent.shape[0], 1, latent.shape[2], latent.shape[3]))
        target = Variable(embedding_indices[:,0,:,:]).long() 

        logits = self(embedding_indices.float())

        loss = self.criterion(logits, target)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        return loss

    def load_model_checkpoint(pixel_model_name, vae_name):
        path = ('checkpoints/' + pixel_model_name + '.ckpt')
        return PixelCNNModule.load_from_checkpoint(path, vae_name = vae_name)