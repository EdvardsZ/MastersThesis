import yaml

config = yaml.load(open('vq_vae.yaml', 'r'), Loader=yaml.FullLoader)

def load_config(model_name):
    return yaml.load(open(model_name + '.yaml', 'r'), Loader=yaml.FullLoader)


def get_model_name(config):
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    data_params = config['data_params']
    model_name = config['model_name']
    
    data_params.update(trainer_params)

    config = data_params
    model_parameter_string = '_'.join([str(value) for key, value in model_params.items()])

    trainer_parameter_string = '&'.join([ str(key) +"="+ str(value) for key, value in config.items()])
    return model_name + '(' + model_parameter_string + ')' + '?' + trainer_parameter_string


