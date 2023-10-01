import yaml

config = yaml.load(open('vq_vae.yaml', 'r'), Loader=yaml.FullLoader)

def load_config(model_name):
    return yaml.load(open(model_name + '.yaml', 'r'), Loader=yaml.FullLoader)

