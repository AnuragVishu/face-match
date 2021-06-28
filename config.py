import os
import configparser
import urllib
from utils.download import download_file_from_google_drive, download_url_to_file

config_path = 'configs/cfg.ini'

config = configparser.ConfigParser()
config.read(config_path)

print(config.sections())

def get_classification_model(model):
    
    model_base_path = config['MODEL_PATH_LOCAL']['ClassificationModelPath']
    model_name = config['MODEL_NAME'][model]
    
    print(f'model base path: {model_base_path}')
    if not os.path.isdir(model_base_path):
        os.mkdir(model_base_path)
    
    model_full_path = os.path.join(model_base_path, model_name)
    
    if os.path.isfile(model_full_path):
        print(f'Model full path: {model_full_path}')
        return model_full_path
    
    if model == 'CelebA':
        print(f'Downloading CelebA model')
        url = config['DOWNLOAD'][model]
        key = url.split('/view')[0].split('d/')[1]
        # file_id = '1X3sRMA-q4nLXdOQ1noqx-TKVzWJkL_kj'
        destination = model_full_path
        try:
            download_file_from_google_drive(key, destination)
        except Exception as e:
            print(f'Exception in downloading model: {e}')
            return None
        return model_full_path

    if model == 'VggFace2':
        print(f'Downloading VggFace2 model')
        url = config['DOWNLOAD'][model]
        try:
            download_url_to_file(url, model_full_path)
        except Exception as e:
            print(f'Exception in downloading model: {e}')
            return None
        return model_full_path
    
    return None

# class Embeddings:



# get_classification_model('VggFace2')