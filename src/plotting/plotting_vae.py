import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_samples_with_reconstruction(model, data_loader, n=6, save_name=None):
    # plot n images and their reconstruction

    x, x_cond, y = next(iter(data_loader))

    image_shape = model.model.image_shape
    model.eval()

    images_to_plot = []

    outputs, z, z_mean, z_log_var = model(x, x_cond, y)

    if isinstance(outputs, list):
        images_to_plot.append(x)
        images_to_plot.append(outputs[0])
        images_to_plot.append(outputs[1])
    else:
        images_to_plot.append(x)
        images_to_plot.append(outputs)

    images_to_plot_titles = ["Original", "Reconstruction", "Reconstruction(Conditioned)"]

    fig = plt.figure(figsize=(n, len(images_to_plot)))
    #fig.suptitle('Figure title')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=len(images_to_plot), ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(images_to_plot_titles[row])

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=n)
        for col, ax in enumerate(axs):
            # plot images
            ax.imshow(images_to_plot[row][col].detach().cpu().reshape(image_shape).permute(1, 2, 0))
            ax.axis('off')
            #ax.set_title(f'Plot title {col}')

    if save_name is not None:
        plt.savefig("assets/reconstructions/" + save_name + ".png")

    plt.show()


def plot_generated_samples(model, n=6, save_name=None):
    # plot n*n generated images
    if model is None or not hasattr(model.model, 'decode'):
        print("Plot generated: Model is None or model has no decoder")
        return
    
    title = "Generated"
    
    model.eval()
    shape = model.model.image_shape
    latent_dim = model.model.latent_dim

    full_image_width = shape[1] * n
    full_image_height = shape[2] * n
    full_image = np.zeros((shape[0], full_image_width, full_image_height))

    for i in range(n):
        for j in range(n):
            z = torch.randn(1, latent_dim)
            image = model.model.decode(z).detach().cpu().numpy().reshape(shape)
            full_image[:, i * shape[1]: (i + 1) * shape[1], j * shape[2]: (j + 1) * shape[2]] = image

    
    full_image = np.transpose(full_image, (1, 2, 0))
    full_image = np.clip(full_image, 0, 1)
    
    plt.title(title)
    plt.imshow(full_image)

    plt.axis('off')
    if save_name is not None:
        plt.savefig("assets/generated/" + save_name + ".png")

    plt.show()







def plot_latent_images(model, n=20, save_name=None):
    # plot n*n images in the latent space
    if model is None or model.model.latent_dim != 2 or not hasattr(model.model, 'decode'):
        #print("Plot latent: Model is None or latent dim is not 2 or model has no decoder")
        return
    
    
    model.eval()

    normal_distribution = torch.distributions.Normal(0, 1)

    grid_x = np.linspace(0.05, 0.95, n)
    grid_y = np.linspace(0.05, 0.95, n)

    image_width = 28
    image_height = 28

    full_image_width = image_width * n
    full_image_height = image_height * n

    full_image = np.zeros((full_image_width, full_image_height))

    for i in range(n):
        for j in range(n):
            z = normal_distribution.icdf(torch.tensor([grid_x[i], grid_y[j]]))
            z = z.view(1, -1).float()
            image = model.model.decode(z).detach().cpu().numpy().reshape(image_width, image_height)
            full_image[i * image_width: (i + 1) * image_width, j * image_height: (j + 1) * image_height] = image

    
    plt.imshow(full_image)

    if save_name is not None:
        plt.savefig("assets/latent/" + save_name + ".png")

    plt.show()