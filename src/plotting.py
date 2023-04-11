import matplotlib.pyplot as plt
import numpy as np
from datasets import get_observation_pixels


def plot_sample_with_conditioned_pixels(example):
    plt.imshow(example)
    # print on top of the image the observation pixels
    obs_x, obs_y = get_observation_pixels()
    for i in range(len(obs_x)):
        plt.text(obs_x[i], obs_y[i], 'X', color='red')
    plt.show()


def plot_samples_with_reconstruction(model, data, n=10, device = 5):
    # plot n images and their reconstruction
    model.eval()

    output = model(data[:n][0].cuda( device = 5), data[:n][1].cuda( device = 5))

    for i in range(n):
        image = data[0][i].detach().cpu().numpy().reshape(28, 28)
        reconstruction = output[0][i].detach().cpu().numpy().reshape(28, 28)


        plt.subplot(2, n, i + 1)
        plt.imshow(image)
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruction)
    plt.show()


    