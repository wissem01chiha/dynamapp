import logging
import math
import jax.numpy as jnp
 
logger = logging.getLogger(__name__)

def wrap_deg(angle):
    """Wrap angle to the interval [-180, 180]."""
    return angle - 360 * jnp.floor((angle + 180) / 360)

def rad_deg(angle_radians):
    """Converts an angle given in radians to degrees."""
    return angle_radians * (180 / math.pi)

def wrap_xpi(angle, X):
    """Wrap angle to the interval [-X*pi, X*pi]."""
    return angle - X * math.pi * jnp.floor((angle + X / 2 * math.pi) / (X * math.pi))

def deg_to_rad(angle_degrees):
    """Converts an angle given in degrees to radians."""
    radians = angle_degrees * (math.pi / 180)
    return radians

def wrap_array(array:jnp.ndarray, lower_bound, upper_bound):
    """wrap the values of an array to the given lower and upper bound """
    return lower_bound + (array - lower_bound)
    
def scale_array(array:jnp.ndarray, lower_bound,upper_bound):
    """Scale the values of an array to the given lower and upper bound """
    min_val = jnp.min(array)
    max_val = jnp.max(array)
    scaled_array = (upper_bound - lower_bound) * (array - min_val) / (max_val - min_val) + lower_bound
    return scaled_array

def clamp_array(array:jnp.ndarray, lower_bound, upper_bound):
    """Clamp the values of an array to the given lower and upper bound """
    clamped_array = jnp.clip(array, lower_bound, upper_bound)
    return clamped_array
