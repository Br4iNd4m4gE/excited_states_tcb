#!/home/cschmidt/miniconda3/envs/excited_states_NN_from_monja/bin/python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from os.path import join, isdir, isfile, dirname, abspath
import os
# sys.path.append("/home/cschmidt/bin/excited_states_networks") # pyNNsMD
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file, extract_number_of_atoms, get_file_length, gaussian, unit_conversions
from pyNNsMD.nn_pes_src.device import set_gpu
import argparse
#from scipy.spatial.distance import pdist
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpuid", type=int)
# ap.add_argument("-p", "--outname") # wird ggf. ignoriert
ap.add_argument("-f", "--file", required=True, help="Path to the input file")
ap.add_argument("-m", "--model", required=False, help="Path to the model", default=None)
ap.add_argument("-s", "--save", action="store_true", help="Save energy and oscillator strength in separate files", default=True)
args = ap.parse_args()
set_gpu([args.gpuid])          ###############  wichtig !!

###################### Define Functions ######################

