from box import Box
import yaml
import os

def get_config(config_path_or_dict: 'str | dict') -> Box:
    """
    Loads a configuration file from a .json or .yaml file, or a dictionary

    Args:
        config_path_or_dict (str | dict): the path to the configuration file or a dictionary

    Returns:
        Box: the configuration file as a Box object
    """

    assert config_path_or_dict, 'Error: No config file provided'
    assert isinstance(config_path_or_dict, str) or isinstance(config_path_or_dict, dict), 'Error: config_path_or_dict must be a path (str) to a .json or .yaml file, or a dict'
    
    if isinstance(config_path_or_dict, dict):
        return Box(config_path_or_dict)
    
    elif isinstance(config_path_or_dict, str):
        assert os.path.exists(config_path_or_dict), f'Error: Config file {config_path_or_dict} does not exist'

        if config_path_or_dict.endswith('.json'):
            return Box.from_json(filename=config_path_or_dict)
        elif config_path_or_dict.endswith('.yaml'):
            return Box.from_yaml(filename=config_path_or_dict, Loader=yaml.FullLoader)
        else:
            raise ValueError('Error: config_path_or_dict must be a path to a .json or .yaml file, or a dict')
        
    else:
        raise ValueError('Error: config_path_or_dict must be a path to a .json or .yaml file, or a dict') # Should never reach this point