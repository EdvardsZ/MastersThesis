from .vae import VAE
from .scvae1d import SCVAE1D
from .scvae2d import SCVAE2D

from .vqvae import VQVAE
from .scvqvae1d import SCVQVAE1D
from .scvqvae2d import SCVQVAE2D

from .base_vae import BaseVAE
from .base_vqvae import BaseVQVAE
from typing import List, Tuple, Type


VAEType = Type[BaseVAE] | Type[BaseVQVAE]
# maybe convert to enum
MODELS_LIST: List[VAEType]= [
            VAE,
            SCVAE1D,
            SCVAE2D,

            VQVAE,
            SCVQVAE1D,
            SCVQVAE2D
]

def get_model(model_name) -> VAEType:
    for model in MODELS_LIST:
        if model.__name__ == model_name:
            return model
    raise ValueError(f"Model {model_name} not found")