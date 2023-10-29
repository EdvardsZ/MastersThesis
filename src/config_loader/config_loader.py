import yaml
import os

def load_config(model_name: str) -> dict:
    return yaml.load(open('configs/' + model_name + '.yaml', 'r'), Loader=yaml.FullLoader)

def get_model_name(config: dict) -> str:
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    data_params = config['data_params']
    model_name = config['model_name']
    
    data_params.update(trainer_params)

    config = data_params
    model_parameter_string = '_'.join([str(value) for key, value in model_params.items()])

    trainer_parameter_string = '&'.join([ str(key) +"="+ str(value) for key, value in config.items()])
    return model_name + '(' + model_parameter_string + ')' + '?' + trainer_parameter_string


def find_all_configs() -> list:
    configs = []
    for file in os.listdir("configs"):
        if file.endswith(".yaml"):
            configs.append(file[:-5])
    return configs