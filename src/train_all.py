from config_loader import find_all_configs, load_config
from train import train_and_evaluate
from datasets.observations import CountSamplingMethod, PixelSamplingMethod

all_configs = find_all_configs()

for config_path in all_configs:
    for dataset in ["MNIST", "CIFAR10", "CelebA"]:
        for count_sampling in [CountSamplingMethod.EXACT, CountSamplingMethod.EXPONENTIAL, CountSamplingMethod.VARIABLE]:
            for pixel_sampling in [PixelSamplingMethod.UNIFORM, PixelSamplingMethod.EXACT, PixelSamplingMethod.GAUSSIAN]:
                if pixel_sampling == PixelSamplingMethod.EXACT and (count_sampling == CountSamplingMethod.VARIABLE or count_sampling == CountSamplingMethod.EXPONENTIAL):
                    continue
                # Pixel sampling does not matter for unconditional models
                config = load_config(config_path)
                config["data_params"]["dataset"] = dataset
                config["data_params"]["count_sampling"] = count_sampling
                config["trainer_params"]["max_epochs"] = 1
                train_and_evaluate(config)
