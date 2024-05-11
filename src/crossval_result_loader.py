import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import lineplot
from model_name import get_config_number, get_model_name, get_pixel_sampling_name, get_method

class Loss:
    def __init__(self, loss_name: str, average: float, std: float):
        self.loss_name = loss_name
        self.average = average
        self.std = std    

class Result:
    def __init__(self, filename: str, data: dict[str, Loss]):
        self.filename = filename
        self.data = data
        
    def get_unconditioned_losses(self) -> list[Loss]:
        
        recon = self.get_unconditioned_reconstruction()
        
        if "VQVAE" in self.filename:
            if "test_vq_loss" in self.data:
                vq = self.data["test_vq_loss"]
                return [ recon, vq]
            else:
                raise Exception("No VQ loss found")
            
        if "VAE" in self.filename:
            if "test_kl_loss" in self.data:
                kl = self.data["test_kl_loss"]
                return [recon, kl]
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
        return get_config_number(self.filename)
    
    def get_model_name(self) -> str:
        return get_model_name(self.filename)
    
    def get_method(self) -> str:
        method = get_method(self.filename)
        if method == "":
            return "-"
        return method
    
    def get_parameters(self) -> str:
        if "VQVAE(" in self.filename or "VAE(" in self.filename:
            return "-"
        bracket_text = self.filename.split("(")[1].split(")")[0]
        name = self.filename.split("(")[0]
        
        res = ""           
        
        res += get_pixel_sampling_name(self.filename)
            
        if "2D" in name:
            if "SOFT" in bracket_text:
                res += ", SoftAdapt"
                
        if "1D" in name:
            exponent = self.filename.split("exponent=")[1].split("&")[0]
            res += f", Exponent={exponent}"
        
        return res
    
    def get_method_name(self) -> str:
        if "VQVAE(" in self.filename or "VAE(" in self.filename:
            return "-"
            
            
        name = self.filename.split("(")[0]
        bracket_text = self.filename.split("(")[1].split(")")[0]
        
        res = ""
        if "SC" in name:
            if "1D" in name:
                res += "Single Decoder"
            else: 
                if "2D" in name:
                    res += "Multi Decoder"
                else:
                    raise Exception("Unknown Decoder method")
        
        if "2D" in name:
            if "SOFT" in bracket_text:
                res += ", SoftAdapt"
                
        pixel_sampling = self.filename.split("pixel_sampling=")[1].split("&")[0]
        if pixel_sampling != "":
            res += f", {pixel_sampling}"
        
        return res
        
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

def get_reduction_factor(file_path: str):
    if "MNIST" in file_path:
        return 128 * 28 * 28
    if "CIFAR10" in file_path:
        return 128 * 32 * 32 * 3
    if "CelebA" in file_path:
        return 128 * 64 * 64 * 3
    raise ValueError("Unknown dataset")

def get_main_columns(df):
    main = []
    for column_0 in df.columns:
        for column_1 in df.columns:
            if column_0 in column_1 and column_0 != column_1 and column_0 not in main and column_0 != "epoch":
                main.append(column_0)
                print(column_0)
    return main
def plot_csv_crossval(df, title, y_label, save_name):
    
    path = "../paper/figures/results/scvae2d/"
    os.makedirs(path, exist_ok=True)
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    for main_column in get_main_columns(df):
        ax = lineplot(data=df, x=df["epoch"], y=main_column, errorbar=None)
        ax.fill_between(df["epoch"], df[main_column + "__MIN"], df[main_column + "__MAX"], alpha=0.2)

    plt.savefig(path + save_name+".pgf")
    plt.show()
    
def drop_unnecessary_columns(df):
    for column in df.columns:
        if "step" in column:
            df = df.drop(column, axis=1)
    return df

def get_method_name(filename) -> str:
    if "VQVAE(" in filename or "VAE(" in filename:
        if "VQVAE" in filename:
            return "VQ-VAE"
        else:
            return "Gaussian VAE"
        
        
    name = filename.split("(")[0]
    bracket_text = filename.split("(")[1].split(")")[0]
    
    res = ""
    if "SC" in name:
        if "1D" in name:
            res += "Single Decoder"
        else: 
            if "2D" in name:
                res += "Multi Decoder"
            else:
                raise Exception("Unknown Decoder method")
    
    if "2D" in name:
        if "SOFT" in bracket_text:
            res += ", SoftAdapt"
            
    pixel_sampling = filename.split("pixel_sampling=")[1].split("&")[0]
    if pixel_sampling != "":
        res += f", {pixel_sampling}"
    
    return res