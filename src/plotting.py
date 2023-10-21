import matplotlib.pyplot as plt
import numpy as np
from datasets import get_observation_pixels
import torch
import torch.nn.functional as F


def plot_sample_with_conditioned_pixels(example):
    plt.imshow(example)
    # print on top of the image the observation pixels
    obs_x, obs_y = get_observation_pixels()
    for i in range(len(obs_x)):
        plt.text(obs_x[i], obs_y[i], 'X', color='red')
    plt.show()


def plot_samples_with_reconstruction(model, data, n=6, save_name=None):
    # plot n images and their reconstruction
    model.eval()

    output = model(data[:n][0], data[:n][1], data[:n][2])

    #make the plot smaller
    plt.figure(figsize=(n, 2))

    for i in range(n):
        image = data[0][i].detach().cpu().numpy().reshape(28, 28)
        if isinstance(output[0], list):
            reconstruction = output[0][0][i].detach().cpu().numpy().reshape(28, 28)
        else:
            reconstruction = output[0][i].detach().cpu().numpy().reshape(28, 28)

        # axis off
        plt.subplot(2, n, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruction)
        plt.axis('off')

    if save_name is not None:
        plt.savefig("assets/reconstructions/" + save_name + ".png")

    plt.show()


def plot_samples_with_reconstruction_and_indices(model, data, n=6, save_name=None):
    # plot n images and their reconstruction
    model.eval()

    output = model(data[:n][0], data[:n][1], data[:n][2])
    indices = output[3].reshape(-1, 1, 7, 7)

    #make the plot smaller
    plt.figure(figsize=(n, 3))

    for i in range(n):
        image = data[0][i].detach().cpu().numpy().reshape(28, 28)
        indice_image = indices[i][0].detach().cpu().numpy().reshape(7, 7)
        reconstruction = output[0][i].detach().cpu().numpy().reshape(28, 28)

        # axis off
        plt.subplot(3, n, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(indice_image)
        plt.axis('off')
        plt.subplot(3, n, i + 1 + 2 *n)
        plt.imshow(reconstruction)
        plt.axis('off')

    if save_name is not None:
        plt.savefig("assets/reconstructions/" + save_name + ".png")


    plt.show()


def plot_latent_images(model, n=20, save_name=None):
    # plot n*n images in the latent space
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
        plt.savefig("assets/generated/" + save_name + ".png")

    plt.show()



def generate_indices_and_reconstruct(model, count = 10, save_name=None):
    vae = model.vae
    sample = torch.Tensor(count, 1, 7, 7)
    sample.fill_(0)
    #Generating images pixel by pixel
    for i in range(7):
        for j in range(7):
            out = model(sample)
            probs = F.softmax(out[:,:,i,j], dim=-1).data
            sample[:,:,i,j] = torch.multinomial(probs, 1).float()
    # plot the images
    #test = sample.view(-1)
    test = sample.long()

    # from {B, 1, 7, 7}
    # 
    # {B 7, 7}
    test = test.squeeze(1)
    
    x_hat = vae.model.reconstruct_from_indices(test, count)

    #FIGURE SIZE
    plt.figure(figsize=(10, 2))
    for i in range(count):
        # show both codes and reconstructions
        plt.subplot(2, count, i + 1)
        plt.imshow(test[i].cpu().numpy())
        plt.axis('off')

        plt.subplot(2, count, i + 1 + count)
        plt.imshow(x_hat[i][0].cpu().numpy())
        plt.axis('off')

    if save_name is not None:
        plt.savefig("assets/generated/" + save_name + ".png")

    plt.show()