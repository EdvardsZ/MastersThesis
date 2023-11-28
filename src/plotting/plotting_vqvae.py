from matplotlib import pyplot as plt
from sympy import plot
from .plotting import plot_sample_images

def plot_samples_with_reconstruction_and_indices(model, data_loader, n=6, save_name=None):
    # plot n images and their reconstruction
    x, x_cond, y = next(iter(data_loader))
    model.eval()
    image_shape = model.model.image_shape

    output = model(x, x_cond, y)

    reconstructions = output[0]
    reconstruction_masked = output[1]
    indices = output[-1].reshape(-1, 1, image_shape[1] // 4, image_shape[2] // 4) ## This is assumpution that downsample that there are 2 downsample layers

    images_to_plot = [x, indices]
    images_to_plot_titles = ["Original", "Indices"]

    for i, reconstruction in enumerate(reconstructions):
        images_to_plot.append(reconstruction)
        images_to_plot_titles.append(f"Reconstruction {i}")

    for i, reconstruction in enumerate(reconstruction_masked):
        if reconstruction is not None:
            images_to_plot.append(reconstruction)
            images_to_plot_titles.append(f"Reconstruction Masked {i}")

    plot_sample_images(images_to_plot, images_to_plot_titles, save_name=save_name, n=n)