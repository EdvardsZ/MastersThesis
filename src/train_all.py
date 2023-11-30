from config_loader import find_all_configs, load_config, get_model_name
from train import train_and_evaluate
from datasets.observations import CountSamplingMethod, PixelSamplingMethod
from copy import deepcopy

def get_sampler_pairs():
    sampler_pairs = []
    for count_sampling in CountSamplingMethod:
        for pixel_sampling in PixelSamplingMethod:
            if pixel_sampling == PixelSamplingMethod.EXACT and (count_sampling == CountSamplingMethod.VARIABLE or count_sampling == CountSamplingMethod.EXPONENTIAL or count_sampling == CountSamplingMethod.POWER_LAW):
                continue
            sampler_pairs.append((count_sampling, pixel_sampling))
    return sampler_pairs


def get_training_configs_for_dataset(dataset):
    sampler_pairs = get_sampler_pairs()
    training_configs = []
    all_configs = find_all_configs()
    for config_path in all_configs:
        config = load_config(config_path)
        model_name = config["model_name"]

        config["data_params"]["dataset"] = dataset

        is_conditioned = "SC" in model_name

        if is_conditioned:
            for count_sampling, pixel_sampling in sampler_pairs:  

                config_copy = deepcopy(config)
                config_copy["data_params"]["count_sampling"] = count_sampling.value
                config_copy["data_params"]["pixel_sampling"] = pixel_sampling.value
                config_copy["trainer_params"]["max_epochs"] = 1

                full_model_name = get_model_name(config_copy)

                print(f"Getting config: {full_model_name}")
                
                training_configs.append(config_copy)

        else:
            config_copy = deepcopy(config)
            config_copy["trainer_params"]["max_epochs"] = 1
            training_configs.append(config_copy)


    return training_configs



for dataset in ["MNIST", "CIFAR10", "CelebA"]:
    print(f"Dataset: {dataset}")
    print("*"*20)
    res = get_training_configs_for_dataset(dataset)
    for config in res:
        full_model_name = get_model_name(config)
        print(f"Training {full_model_name}")
        train_and_evaluate(config)