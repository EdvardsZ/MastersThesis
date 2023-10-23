import math
import torch
def get_observation_pixels(image_shape):
    
    image_size= image_shape[1]

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

def get_random_observation_pixels(image_shape):

    obs_x = []
    obs_y = []

    rand = torch.randperm(image_shape[1] * image_shape[2])[:16]
    
    for i in range(16):
        obs_x.append(int(rand[i] / image_shape[2]))
        obs_y.append(rand[i]% image_shape[2])
    return obs_x, obs_y
