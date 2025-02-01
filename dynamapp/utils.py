import logging
import math
import json
import numpy as np
import yaml

logger = logging.getLogger(__name__)

def wrap_deg(angle):
    """Wrap angle to the interval [-180, 180]."""
    return angle - 360 * np.floor((angle + 180) / 360)

def rad_deg(angle_radians):
    """Converts an angle given in radians to degrees."""
    return angle_radians * (180 / math.pi)

def wrap_xpi(angle, X):
    """Wrap angle to the interval [-X*pi, X*pi]."""
    return angle - X * math.pi * np.floor((angle + X / 2 * math.pi) / (X * math.pi))

def deg_to_rad(angle_degrees):
    """Converts an angle given in degrees to radians."""
    radians = angle_degrees * (math.pi / 180)
    return radians

def wrap_array(array:np.ndarray, lower_bound, upper_bound):
    """wrap the values of an array to the given lower and upper bound """
    return lower_bound + (array - lower_bound)
    
def scale_array(array:np.ndarray, lower_bound,upper_bound):
    """Scale the values of an array to the given lower and upper bound """
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (upper_bound - lower_bound) * (array - min_val) / (max_val - min_val) + lower_bound
    return scaled_array

def clamp_array(array:np.ndarray, lower_bound, upper_bound):
    """Clamp the values of an array to the given lower and upper bound """
    clamped_array = np.clip(array, lower_bound, upper_bound)
    return clamped_array

def dict_to_json(structData, filename):
    """Saves a Python dictionary to a JSON file."""
    if not isinstance(structData, dict):
        logger.error('The first input argument must be a Python dictionary')
    if not isinstance(filename, str) or filename == '':
        logger.error('The second input argument must ba non-empty string representing the filename')
    try:
        with open(filename, 'w') as file:
            json.dump(structData, file, indent=4)
    except IOError:
        logger.error(f'Could not create or open the file "{filename}" for writing')

def yaml_to_dict(yamlFilePath) -> dict:
    """
    Get parameters from the config YAML file and return them as a 
    dictionary.
    
    Args:
        yamlFilePath (str): Path to the YAML file.
    """
    try:
        with open(yamlFilePath, 'r') as file:
            dic = yaml.safe_load(file)
        return dic
    except FileNotFoundError:
        logger.error(f"Error: File '{yamlFilePath}' not found.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{yamlFilePath}': {e}")
        return {}