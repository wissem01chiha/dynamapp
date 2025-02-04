"""
Setup file for package tests.
==============================
"""
import os
import sys
import copy
import unittest
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt 
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
