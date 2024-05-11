

def get_config_number(model_name) -> int:
        bracket_text = model_name.split("(")[1].split(")")[0]
        if "VQVAE" in model_name:
            embedding_dim = bracket_text.split("_")[1]
            
            if embedding_dim == "16":
                return 1
            if embedding_dim == "32":
                return 2
            if embedding_dim == "64":
                return 3
            else :
                raise Exception("Unknown embedding_dim:" + embedding_dim)
        if "VAE" in model_name:
            latent_dim = bracket_text.split("_")[0]
            if latent_dim == "16":
                return 1
            if latent_dim == "64":
                return 2
            else:
                raise Exception("Unknown bracket_text:" + latent_dim)
        raise Exception("Unknown model type")
    
def get_model_name(model_name) -> str:
    if "VQVAE" in model_name:
        return "VQ-VAE"
    else:
        return "Gaussian VAE"
    
def get_pixel_sampling_name(model_name: str) -> str:
    pixel_sampling = model_name.split("pixel_sampling=")[1].split("&")[0]
    if pixel_sampling != "":
        if pixel_sampling == "UNIFORM":
            return "Uniform sampling"
        if pixel_sampling == "GAUSSIAN":
            return "Gaussian sampling"
        if pixel_sampling == "EXACT":
            return "Exact sampling"
        else:
            raise Exception("Unknown pixel_sampling:" + pixel_sampling)
    raise Exception("No pixel_sampling in model_name")

def get_method(model_name) -> str:
    if "VQVAE(" in model_name or "VAE(" in model_name:
        return ""
    name = model_name.split("(")[0]
    if "SC" in name:
        if "1D" in name:
            return "Single Decoder"
        else: 
            if "2D" in name:
                return "Multi Decoder"
            else:
                raise Exception("Unknown Decoder method")   
    raise Exception("Unknown model type")