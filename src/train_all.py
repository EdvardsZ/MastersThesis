from config import find_all_configs, load_config
from train import train_and_evaluate

all_configs = find_all_configs()

for config_path in all_configs:
    for dataset in ["MNIST", "FashionMNIST"]: # TODO ADD more datasets
        config = load_config(config_path)
        config["data_params"]["dataset"] = dataset
        train_and_evaluate(config)
