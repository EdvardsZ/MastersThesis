from .plotting_vae import plot_generated_samples, plot_samples_with_reconstruction, plot_latent_images
from .plotting_vqvae import plot_samples_with_reconstruction_and_indices
from models import BaseVAE, BaseVQVAE
from loss import VAELoss
from modules import VAEModule
def plot_stage_one_results(model: VAEModule, data_loader, save_name = None):
    if isinstance(model.model, BaseVAE):
        plot_samples_with_reconstruction(model, data_loader, save_name=save_name)
        plot_latent_images(model, save_name=save_name)
        plot_generated_samples(model, save_name=save_name)
    else:
        # VQ-VAES
        plot_samples_with_reconstruction_and_indices(model, data_loader, save_name=save_name)
    return



