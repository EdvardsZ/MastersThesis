from .plotting_vae import plot_generated_samples, plot_samples_with_reconstruction, plot_latent_images
def plot_stage_one_results(model, data_loader, save_name = None):
    plot_samples_with_reconstruction(model, data_loader, save_name=save_name)
    plot_latent_images(model, save_name=save_name)
    plot_generated_samples(model, save_name=save_name)



