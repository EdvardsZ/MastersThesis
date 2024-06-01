from cProfile import label
import os
import re
from crossval_result_loader import get_results, Result, get_reduction_factor
from seaborn import lineplot
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

dataset = "CIFAR10"
project_name = "MTVAEs_05.01-cross-val"
directory = f"assets/results/raw/{project_name}/{dataset}/"

results: list[Result] = get_results(directory)

def create_plot_reconstruction_results(gaussian: list[Result], uniform: list[Result], baseline: Result | None = None):
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots()
    
    exponential_values: list[float] = []
    
    line_gaussian = []
    line_gaussian_min = []
    line_gaussian_max = []

    sorted_gaussian = sorted(gaussian, key=lambda x: x.get_exponential_value())
    for result in sorted_gaussian:
        exponential = result.get_exponential_value()
        reconstruction_loss = result.get_unconditioned_losses()[0]
        average = reconstruction_loss.average
        std = reconstruction_loss.std
            
        
            
        if baseline is None:
            average = average / (128 * 32 * 32 * 3)
            std = average / (128 * 32 * 32 * 3)
            
            
        line_gaussian.append(average)
        line_gaussian_min.append(average - std)
        line_gaussian_max.append(average - std)
        
        exponential_values.append(exponential)
        print(result.get_exponential_value())
    

        
    #convert to df the whole thing
    df = pd.DataFrame({
        "exponential_values": exponential_values,
        "line_gaussian": line_gaussian,
        "line_gaussian_min": line_gaussian_min,
        "line_gaussian_max": line_gaussian_max,
    })
        
    plt.title("Reconstruction Loss(Masked) vs Exponential Value")
    plt.xlabel("Exponential Value")
    plt.ylabel("Reconstruction Loss")
    
    # create lineplot and fill_between
    ax = lineplot(data=df, x=df["exponential_values"], y=df["line_gaussian"], errorbar=None, label="Gaussian sampling")
    
    #plot baseline as a horizontal line
    if baseline is not None:
        ax = ax.axhline(y=baseline.get_unconditioned_losses()[0].average, color='r', linestyle='--', label="Baseline")
    
    if baseline is not None:
        save_path = f"../paper/figures/results/{dataset}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}reconstruction_loss_vs_exponential_value_vqvae.pgf")
    else:
        save_path = f"../paper/figures/results/{dataset}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}reconstruction_loss_vs_exponential_value_gaussian.pgf")
        
    plt.show()
    
    

def plot_exponential_value_results_VQ(res: list[Result]):
    
    
    
    result = list(filter(lambda x: "VQ-VAE" in x.get_model_name() and "-" in x.get_method() and x.get_config_number() == 2, results))[0]
    
    res = list(filter(lambda x: "VQ-VAE" in x.get_model_name() and x.get_config_number() == 2, results))
    res = list(filter(lambda x: "Single" in x.get_method(), res))
    
    gaussian = list(filter(lambda x: "GAUSSIAN" in x.filename, res))
    unifrom = list(filter(lambda x: "UNIFORM" in x.filename, res))
    
    create_plot_reconstruction_results(gaussian, unifrom, result)
    
def plot_exponential_value_results_Gaussian(res: list[Result]):
    
    res = list(filter(lambda x: "Gaussian VAE" in x.get_model_name() and x.get_config_number() == 2, results))
    res = list(filter(lambda x: "Single" in x.get_method(), res))
    
    gaussian = list(filter(lambda x: "GAUSSIAN" in x.filename, res))
    unifrom = list(filter(lambda x: "UNIFORM" in x.filename, res))
    
    create_plot_reconstruction_results(gaussian, unifrom)

plot_exponential_value_results_VQ(results)
plot_exponential_value_results_Gaussian(results)
    