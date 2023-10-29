import matplotlib.pyplot as plt
from sympy import plot
import torch
import torchvision.transforms as transforms

from datasets import PartialObservation



def plot_conditioned_examples(example: torch.Tensor):
    plot_conditioned_example(example, "exact")
    plot_conditioned_example(example, "random")


def plot_conditioned_example(example: torch.Tensor, conditioning_mode: str, save = False):
    partial_observation = PartialObservation(conditioning_mode, add_mask = True)

    plt.subplot(1, 3, 1)
    reshaped = example.detach().cpu().numpy().reshape(example.shape[1], example.shape[2])
    plt.imshow(reshaped)
    plt.axis('off')

    x_cond = partial_observation.get_partial_observation(example)

    plt.subplot(1, 3, 2)
    reshaped = x_cond[0].detach().cpu().numpy().reshape(example.shape[1], example.shape[2])
    plt.imshow(reshaped)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    reshaped = x_cond[1].detach().cpu().numpy().reshape(example.shape[1], example.shape[2])
    plt.imshow(reshaped)
    plt.axis('off')

    #save image
    if save:
        plt.savefig(f"assets/observations/{conditioning_mode}.png")

    plt.show()