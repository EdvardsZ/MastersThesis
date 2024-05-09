import torch
import os

class Loss:
    def __init__(self, loss_name: str, average: float, std: float):
        self.loss_name = loss_name
        self.average = average
        self.std = std    

class Result:
    def __init__(self, filename: str, data: dict[str, Loss]):
        self.filename = filename
        self.data = data
        
    def get_unconditioned_losses(self) -> dict[str, Loss]:
        
        recon = self.get_unconditioned_reconstruction()
        
        if "VQVAE" in self.filename:
            if "test_vq_loss" in self.data:
                vq = self.data["test_vq_loss"]
                return {"Reconstruction loss": recon, "VQ objective loss": vq}
            else:
                raise Exception("No VQ loss found")
            
        if "VAE" in self.filename:
            if "test_kl_loss" in self.data:
                kl = self.data["test_kl_loss"]
                return {"Reconstruction loss": recon, "KL divergence loss": kl}
            else:
                raise Exception("No KL loss found")
            
        raise Exception("Unknown model type")       
            
    def get_unconditioned_reconstruction(self) -> Loss:
        if "test_recon_loss_0(MASKED)" in self.data:
            recon = self.data["test_recon_loss_0(MASKED)"]
        if "test_recon_loss_0" in self.data:
            recon = self.data["test_recon_loss_0"]
            
        if recon is None:
            raise Exception("No recon loss found")
        
        return recon      
        
    def get_config_number(self) -> int:
        bracket_text = self.filename.split("(")[1].split(")")[0]
        if "VQVAE" in self.filename:
            embedding_dim = bracket_text.split("_")[1]
            
            if embedding_dim == "16":
                return 1
            if embedding_dim == "32":
                return 2
            if embedding_dim == "64":
                return 3
            else :
                raise Exception("Unknown embedding_dim:" + embedding_dim)
        if "VAE" in self.filename:
            latent_dim = bracket_text.split("_")[0]
            if latent_dim == "16":
                return 1
            if latent_dim == "64":
                return 2
            else:
                raise Exception("Unknown bracket_text:" + latent_dim)
        raise Exception("Unknown model type")
        
    def get_display_model_name(self) -> str:
        name = self.filename.split("(")[0]
        
        res = ""
        if "VQ" in name:
            res += "VQ-VAE"
        else:
            res += "Gaussian VAE"
            
        config_number = self.get_config_number()
        
        res += f"(Conf. Nr.{config_number})"
        
        if "SC" in name:
            if "1D" in name:
                res += " with Single Decoder method"
            else: 
                if "2D" in name:
                    res += " with Multi Decoder method"
                else:
                    raise Exception("Unknown Decoder method")
        
        return res        
    
        
def get_crossval_results(data: list)-> dict[str, Loss]:
    losses: dict[str, Loss]= {}
    
    first : dict = data[0][0]
    
    keys = first.keys()
    
    for key in keys:
        values = []
        for item in data:
            values.append(item[0][key])
            
        average = sum(values) / len(values)
        std = sum([(value - average)**2 for value in values]) / len(values)
        
        losses[key] = Loss(key, average, std)
        
    return losses

def get_results(directory: str) -> list[Result]:
    results = []
    print("Loading results...")
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            # load the file
            data = torch.load(directory + filename)
            data = get_crossval_results(data)
            results.append(Result(filename, data))
            #print(f"Loaded {filename}")
            
    print("Loaded: ", len(results))
    print("Done.")
    return results


def check_if_done_already(full_model_name, project_name) -> bool:
    # find in the model name the dataset name ( 'dataset=DATASET_NAME&)
    dataset_name = full_model_name.split("dataset=")[1].split("&")[0]
    path = f"assets/results/raw/{project_name}/{dataset_name}/{full_model_name}_crossval_results.pt"
    
    if os.path.exists(path):
        return True
    
    return False

def check_if_pending(full_model_name, project_name) -> bool:
    # find in the model name the dataset name ( 'dataset=DATASET_NAME&)
    dataset_name = full_model_name.split("dataset=")[1].split("&")[0]
    path = f"assets/results/pending/{project_name}/{dataset_name}/{full_model_name}_crossval_results.txt"
    
    if os.path.exists(path):
        return True
    
    return False

def mark_as_pending(full_model_name, project_name, job_nr: int):
    # find in the model name the dataset name ( 'dataset=DATASET_NAME&)
    dataset_name = full_model_name.split("dataset=")[1].split("&")[0]
    path = f"assets/results/pending/{project_name}/{dataset_name}/{full_model_name}_crossval_results.txt"
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Pending: " + str(job_nr))
        
    return

def remove_pending(full_model_name, project_name):
    # find in the model name the dataset name ( 'dataset=DATASET_NAME&)
    dataset_name = full_model_name.split("dataset=")[1].split("&")[0]
    path = f"assets/results/pending/{project_name}/{dataset_name}/{full_model_name}_crossval_results.txt"
    
    os.remove(path)
    return