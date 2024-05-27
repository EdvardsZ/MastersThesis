import matplotlib.pyplot as plt
from datasets import load_dataset

sample_data_config = {
    "dataset": "CelebA",
    "batch_size": 32,
    "count_sampling": "EXACT",
    "pixel_sampling": "EXACT"
}

train_loader, val_loader, test_loader, image_shape = load_dataset(sample_data_config)