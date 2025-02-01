"""
Setup file for package tests.
==============================
"""
import os
import sys
import copy
import unittest
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
