
from config_loader import get_training_configs_for_dataset, print_model_params, get_model_name

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

for dataset in ["MNIST","CIFAR10", "CelebA"]: #, "CIFAR10", "CelebA"]:
    print(f"Dataset: {dataset}")
    print("*"*20)
    
    res = get_configs_sorted(dataset)

    print_summary(res)
    
    for config in res:
        full_model_name = get_model_name(config)
        print(f"Training {full_model_name}")
        train_and_evaluate(config)