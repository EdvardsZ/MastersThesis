from .lin_vae import *
from .vae import *
from .label_cond_vae import *
from .label_mdvae import *
from .pixel_cond_vae import *
from .pixel_mdvae import *
from .vq_vae import *
from .simple_vq_vae import *

MODELS_LIST = [ VQVAE, VAE]


def get_model(model_name):
    for model in MODELS_LIST:
        if model.__name__ == model_name:
            return model
    raise ValueError(f"Model {model_name} not found")