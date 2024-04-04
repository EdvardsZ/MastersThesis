from .config_loader import find_all_configs, load_config, get_model_name, print_model_params
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
        config["trainer_params"]["max_epochs"] = 100

        is_conditioned = "SC" in model_name
        is_second_method = "2" in model_name
        

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
            copy = deepcopy(config)
            training_configs.append(copy)

        #print_model_params(config)
        samplers = "[" + ", ".join([f"({count.value}, {pixel.value})" for count, pixel in sampler_pairs]) + "]"
        #print("Sampling pairs: ", samplers)

    return training_configs