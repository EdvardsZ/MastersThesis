from config_loader import find_all_configs, load_config
from train import train_and_evaluate

all_configs = find_all_configs()

for config_path in all_configs:
    for dataset in ["MNIST", "CIFAR10"]: # TODO ADD more datasets
        config = load_config(config_path)
        config["data_params"]["dataset"] = dataset
        config["trainer_params"]["max_epochs"] = 1
        train_and_evaluate(config)
