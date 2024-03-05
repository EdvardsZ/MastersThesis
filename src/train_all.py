from random import sample
from config_loader import find_all_configs, load_config, get_model_name, print_model_params
from train import train_and_evaluate
from datasets.observations import CountSamplingMethod, PixelSamplingMethod
from copy import deepcopy

def get_sampling_pairs_no_conditioning():
    return [(CountSamplingMethod.EXACT, PixelSamplingMethod.EXACT)]

def get_sampling_pairs_single_decoder():
    sampler_pairs = []

    count_sampling = CountSamplingMethod.POWER_LAW

    sampler_pairs.append((count_sampling, PixelSamplingMethod.GAUSSIAN))
    sampler_pairs.append((count_sampling, PixelSamplingMethod.UNIFORM))

    return sampler_pairs

def get_sampling_pairs_multi_decoder():
    sampler_pairs = []

    count_sampling = CountSamplingMethod.EXACT

    sampler_pairs.append((count_sampling, PixelSamplingMethod.EXACT))
    sampler_pairs.append((count_sampling, PixelSamplingMethod.UNIFORM))

    return sampler_pairs

def apply_sampling_pairs_to_config(config, count_sampling, pixel_sampling):
    config_copy = deepcopy(config)
    config_copy["data_params"]["count_sampling"] = count_sampling.value
    config_copy["data_params"]["pixel_sampling"] = pixel_sampling.value

    return config_copy



def get_training_configs_for_dataset(dataset):
    training_configs = []
    all_configs = find_all_configs()
    for config_path in all_configs:
        config = load_config(config_path)
        model_name = config["model_name"]

        config["data_params"]["dataset"] = dataset
        config["trainer_params"]["max_epochs"] = 1


        is_conditioned = "SC" in model_name
        is_second_method = "2" in model_name
        

        if not is_conditioned: 
            training_configs.append(config)

        sampler_pairs = []
        
        if is_conditioned:
            if is_second_method:

                sampler_pairs = get_sampling_pairs_multi_decoder()
                for count_sampling, pixel_sampling in sampler_pairs:

                    copy = apply_sampling_pairs_to_config(config, count_sampling, pixel_sampling)
                    training_configs.append(copy)

            if not is_second_method:
                sampler_pairs = get_sampling_pairs_single_decoder()
                for count_sampling, pixel_sampling in sampler_pairs:
                    copy = apply_sampling_pairs_to_config(config, count_sampling, pixel_sampling)
                    training_configs.append(copy)
        else:
            sampler_pairs = get_sampling_pairs_no_conditioning()
            for count_sampling, pixel_sampling in sampler_pairs:
                copy = apply_sampling_pairs_to_config(config, count_sampling, pixel_sampling)
                training_configs.append(copy)

        #print_model_params(config)
        samplers = "[" + ", ".join([f"({count.value}, {pixel.value})" for count, pixel in sampler_pairs]) + "]"
        #print("Sampling pairs: ", samplers)

    return training_configs


for dataset in ["MNIST"]: #, "CIFAR10", "CelebA"]:
    print(f"Dataset: {dataset}")
    print("*"*20)
    res = get_training_configs_for_dataset(dataset)

    # sort by if the model is VQ or not and then by the if it is conditioned or not and then by if it is the second method or not
    res = sorted(res, key=lambda x: (not "VQ" in x["model_name"], "SC" in x["model_name"], "2" in x["model_name"]))

    print(f"Total number of configs: {len(res)}")

    for config in res:
        print_model_params(config)
        print(config["data_params"]["count_sampling"], config["data_params"]["pixel_sampling"])

    print("*"*20)
    print("Ready to train")
    for config in res:
        full_model_name = get_model_name(config)
        print(f"Training {full_model_name}")
        train_and_evaluate(config)