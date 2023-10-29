import math
import torch
from typing import Tuple, List
from PIL import Image
import torchvision.transforms as transforms

class PartialObservation:
    def __init__(self, conditioning_mode: str):
        self.conditioning_mode = conditioning_mode

    def get_partial_observation(self, image: torch.Tensor) -> torch.Tensor:
        data = self.image_to_tensor(image)
        self.image_shape = data.shape

        obs_x, obs_y = self.get_observation_pixels()

        if self.conditioning_mode == "exact":
            cond_data = torch.zeros_like(data)
            cond_data[:, obs_x, obs_y] = data[:, obs_x, obs_y]
            return cond_data
        if self.conditioning_mode == "random": 
            cond_data = torch.zeros((data.shape[0] + 1, data.shape[1], data.shape[2]))
            cond_data[1:, obs_x, obs_y] = data[:, obs_x, obs_y]
            cond_data[0, obs_x, obs_y] = 1
            return cond_data
        raise ValueError("conditioning_mode must be exact or random!!!")

    def get_observation_pixels(self) -> Tuple[List[int], List[int]]:
        if self.conditioning_mode == "exact":
            return self.get_observation_exact_pixels()
        if self.conditioning_mode == "random":
            return self.get_random_observation_pixels()
        raise ValueError("conditioning_mode must be exact or random!!!")
    

    def get_observation_exact_pixels(self) -> Tuple[List[int], List[int]]:
        
        image_size= self.image_shape[1]

        observation_rows = 4
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

    def get_random_observation_pixels(self) -> Tuple[List[int], List[int]]:
        obs_x = []
        obs_y = []

        rand = torch.randperm(self.image_shape[1] * self.image_shape[2])[:16]
        
        for i in range(16):
            obs_x.append(int(rand[i] / self.image_shape[2]))
            obs_y.append(rand[i]% self.image_shape[2])
        return obs_x, obs_y
    
    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = transforms.ToPILImage()(image)
        data = transforms.ToTensor()(image)
        return data
