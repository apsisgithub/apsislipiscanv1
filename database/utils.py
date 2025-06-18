
import yaml
from loguru import logger
import sys

def config_loader(config_file: str) -> dict:
    """load yaml file and return yaml files parameter name and values in dictionay format.

    Args:
        config_file (str): yaml config file location

    
    Returns:
        dict: yaml file parameter name and values in dictionay format
    """
    
    conf = None
    # load config yaml file
    try:
        with open(config_file, "r") as stream:
            try:
                conf = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
                sys.exit(-1)
    
    except FileNotFoundError as f_error:
        logger.error(f_error)
        sys.exit(-1)
    
    return conf