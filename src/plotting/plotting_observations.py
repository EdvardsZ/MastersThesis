import matplotlib.pyplot as plt
from sympy import plot
import torch
import torchvision.transforms as transforms
from datasets import PartialObservation
from datasets.observations import CountSamplingMethod, PixelSamplingMethod


def plot_conditioned_examples(example: torch.Tensor):
    count_sampling = CountSamplingMethod.EXACT
    plot_conditioned_example(example, count_sampling, PixelSamplingMethod.EXACT)
    plot_conditioned_example(example, count_sampling, PixelSamplingMethod.UNIFORM)
    plot_conditioned_example(example, count_sampling, PixelSamplingMethod.GAUSSIAN)


def plot_conditioned_example(example: torch.Tensor, count_sampling: CountSamplingMethod, pixel_sampling: PixelSamplingMethod, save = True):
    partial_observation = PartialObservation(count_sampling, pixel_sampling, add_mask = True)

    fig = plt.figure(figsize=(6, 2))
    fig.suptitle(f"Count sampling: {count_sampling.name}, Pixel sampling: {pixel_sampling.name}")

    shape = example.shape
    plt.subplot(1, 3, 1)
    reshaped = example.detach().cpu().reshape(shape).permute(1, 2, 0)
    plt.imshow(reshaped)
    plt.axis('off')

    x_cond = partial_observation.get_partial_observation(example)

    plt.subplot(1, 3, 2)
    reshaped = x_cond[-1].detach().cpu().reshape((1, shape[1], shape[2])).permute(1, 2, 0)
    plt.imshow(reshaped)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    reshaped = x_cond[:-1].detach().cpu().reshape(shape).permute(1, 2, 0)
    plt.imshow(reshaped)
    plt.axis('off')

    #save image
    if save:
        plt.savefig(f"assets/observations/{pixel_sampling.name}.png")

    plt.show()