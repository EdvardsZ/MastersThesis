import torch
from .pixel_sampler import PixelSampler
from typing import Tuple, List
import math

class ExactPixelSampler(PixelSampler):
    def __init__(self, add_mask: bool = False):
        self.add_mask = add_mask

    def sample(self, image: torch.Tensor, pixel_count) -> torch.Tensor:
        self.image_shape = image.shape

        obs_x, obs_y = self.get_observation_exact_pixels(image)
        if self.add_mask == False or self.add_mask is None:
            cond_data = torch.zeros_like(image)
            cond_data[:, obs_x, obs_y] = image[:, obs_x, obs_y]
            return cond_data
        else:
            cond_data = torch.zeros((image.shape[0] + 1, image.shape[1], image.shape[2]))
            cond_data[:image.shape[0], obs_x, obs_y] = image[:, obs_x, obs_y]
            cond_data[-1, obs_x, obs_y] = 1
            return cond_data
        
    def get_observation_exact_pixels(self, image: torch.Tensor) -> Tuple[List[int], List[int]]:
        image_size= image.shape[1]
        observation_rows = image.shape[1] // 7
        spacing = math.ceil(((image_size - observation_rows) / (observation_rows + 1)))
        without_sides = ((spacing * (observation_rows - 1)) + observation_rows)
        sides = (image_size - without_sides)
        start = int(sides / 2) + 1
        stop = int(image_size  - (sides - start))

        obs_x_n=int(spacing)
        obs_y_n=int(spacing)

        obs_x=[]
        obs_y=[]
        for i in range(start,stop,obs_x_n):
            for j in range(start,stop,obs_y_n):
                obs_x.append(i)
                obs_y.append(j)
        return obs_x, obs_y