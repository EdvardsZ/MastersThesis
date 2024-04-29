
from config_loader import get_training_configs_for_dataset, print_model_params, get_model_name
from train import train_and_evaluate
from loss.adapt import AdaptiveMode
from copy import deepcopy

def print_summary(configs):
    print(f"Total number of configs: {len(configs)}")
    
    for config in configs:
        print_model_params(config)
        try:
            print(config["data_params"]["count_sampling"], config["data_params"]["pixel_sampling"])
        except:
            # do nothign
            pass

    print("*"*20)
    print("Ready to train")
    
def get_configs_sorted(dataset):
    res = get_training_configs_for_dataset(dataset)
    # sort by if the model is VQ or not and then by the if it is conditioned or not and then by if it is the second method or not
    res = sorted(res, key=lambda x: (not "VQ" in x["model_name"], "SC" in x["model_name"], "2" in x["model_name"]))
    
    return res

def create_copy_of_config_with_adaptive_mode(config, param):
    config_copy = deepcopy(config)
    config_copy["model_params"]["adaptive_mode"] = param.value
    return config_copy

def create_copy_of_config_with_exponent_and_power_law(config, exponent):
    config_copy = deepcopy(config)
    config_copy["data_params"]["count_sampling"] = "POWER_LAW"
    config_copy["data_params"]["exponent"] = exponent
    return config_copy

def add_extra_configs(configs):
    res = []
    
    for config in configs:
        
        if "2D" in config["model_name"]:
            # here play around with the adaptive mode
            copy = create_copy_of_config_with_adaptive_mode(config, AdaptiveMode.SOFT)
            res.append(copy)
            copy = create_copy_of_config_with_adaptive_mode(config, AdaptiveMode.SCALED)
            res.append(copy)
               
        if "1D" in config["model_name"]:
            # here play around with different exponential values
            copy = create_copy_of_config_with_exponent_and_power_law(config, 60)
            res.append(copy)  
            
            copy = create_copy_of_config_with_exponent_and_power_law(config, 50)
            res.append(copy)    
            
            copy = create_copy_of_config_with_exponent_and_power_law(config, 30)
            res.append(copy)
            
            copy = create_copy_of_config_with_exponent_and_power_law(config, 20)
            res.append(copy)
            
            
            config["data_params"]["exponent"] = 40
            
        res.append(config)
        
    return res


for dataset in ["MNIST", "CIFAR10", "CelebA"]:
    print(f"Dataset: {dataset}")
    print("*"*20)
    
    configs = get_configs_sorted(dataset)
    configs = add_extra_configs(configs)
    
    print_summary(configs)
    
    for config in configs:
        full_model_name = get_model_name(config)
        print(f"Training {full_model_name}")
        train_and_evaluate(config)