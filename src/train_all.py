from config import find_all_configs
from train import train_and_evaluate

all_configs = find_all_configs()

for config in all_configs:
    train_and_evaluate(config)
