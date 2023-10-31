from .vae import VAE
from .scvae1d import SCVAE1D
from .scvae2d import SCVAE2D


# maybe convert to enum
MODELS_LIST = [ VAE,
                SCVAE1D,
                SCVAE2D
                  ]

def get_model(model_name):
    for model in MODELS_LIST:
        if model.__name__ == model_name:
            return model
    raise ValueError(f"Model {model_name} not found")