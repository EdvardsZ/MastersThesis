import yaml
import os

def load_config(model_name: str) -> dict:
    file_name = model_name + '.yaml'
    
    for folder in os.listdir('configs'):
        if os.path.isfile('configs/' + folder + '/' + file_name):
            return yaml.load(open('configs/' + folder + '/' + file_name, 'r'), Loader=yaml.FullLoader)
        
    raise FileNotFoundError('Config file not found')

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
    for folder in os.listdir("configs"):
        if(os.path.isdir("configs/" + folder)):
            for file in os.listdir("configs/" + folder):
                if(file.endswith(".yaml")):
                    configs.append(file[:-5])
    return configs