from .vae import VAE


# maybe convert to enum
MODELS_LIST = [ VAE ]

def get_model(model_name):
    for model in MODELS_LIST:
        if model.__name__ == model_name:
            return model
    raise ValueError(f"Model {model_name} not found")