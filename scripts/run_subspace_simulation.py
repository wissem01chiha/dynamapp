import argparse
import sys
import os
import numpy as np 
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt 
import logging
import time

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description=\
'simulate manipulator subspace identification models')

parser.add_argument('--v',type=bool,default=False)
parser.add_argument('--cutoff_frequency', type=float, default=3)
parser.add_argument('--show_figures', type=bool,default=False)
parser.add_argument('--data_file',type=str,default='blast_traj.csv')
parser.add_argument('--filter',type=bool,default=True)
args = parser.parse_args()

""" 
https://www.researchgate.net/publication/224627792
Pole Placement of Time-Varying State Space Representations
"""