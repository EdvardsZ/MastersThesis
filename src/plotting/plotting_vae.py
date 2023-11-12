import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_samples_with_reconstruction(model, data_loader, n=6, save_name=None):
    # plot n images and their reconstruction

    x, x_cond, y = next(iter(data_loader))

    image_shape = x.shape[1:]
    print(image_shape)
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
            ax.imshow(images_to_plot[row][col].detach().cpu().numpy().reshape(image_shape).squeeze())
            ax.axis('off')
            #ax.set_title(f'Plot title {col}')

    if save_name is not None:
        plt.savefig("assets/reconstructions/" + save_name + ".png")

    plt.show()

def plot_latent_images(model, n=20, save_name=None):
    # plot n*n images in the latent space
    if model is None or model.model.latent_dim != 2 or not hasattr(model.model, 'decode'):
        print("Model is None or latent dim is not 2 or model has no decoder")
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