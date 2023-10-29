import yaml

def load_config(model_name: str) -> dict:
    return yaml.load(open('configs/' + model_name + '.yaml', 'r'), Loader=yaml.FullLoader)