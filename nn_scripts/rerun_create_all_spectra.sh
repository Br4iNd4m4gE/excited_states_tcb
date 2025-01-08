#!/home/cschmidt/miniconda3/envs/excited_states_NN_from_monja/bin/python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from os.path import join, isdir, isfile, dirname, abspath
import os
# sys.path.append("/home/cschmidt/bin/excited_states_networks") # pyNNsMD
sys.path.append(abspath(join(dirname(__file__), "..")))
from pyNNsMD.utils.general import parse_single_file
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
from ase import Atoms
from ase.io import write
import shutil
import logging

ap = argparse.ArgumentParser()
# ap.add_argument("-g", "--gpuid", type=int)
# ap.add_argument("-p", "--outname") # wird ggf. ignoriert
ap.add_argument("model", help="Path to the model")
args = ap.parse_args()
# set_gpu([args.gpuid])          ###############  wichtig !!

###################### Define Functions ######################

def gaussian(x, mean, stddev):
    return 1 / (stddev * np.sqrt(2*np.pi) ) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def gaussian_A1(x, mean, stddev):
    return np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def get_file_length(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    # The output will be in the form "number file_path", so we split it and take the first part
    line_count = int(result.stdout.split()[0])
    return line_count

def extract_number_of_atoms(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_atoms = 0
    for line in lines:
        # Count the number of lines until the first empty line
        line_tmp = line.strip()
        if line_tmp == "":
            break
        num_atoms += 1
    
    # num_atoms -= 1 # Subtract one because the first line is the energy
    return num_atoms

def multiples_of_n_in_between_interval(interval_bottom, interval_top, n):
    interval = [interval_bottom, interval_top]
    min_val, max_val = np.min(interval), np.max(interval)
    # Step 1: Calculate the next higher multiple of n for the smallest element
    next_multiple_min = n * np.ceil(min_val / n)

    # Step 2: Calculate the highest multiple of n for the largest element
    next_multiple_max = n * np.floor(max_val / n)

    # Step 3: Generate the new array with values spaced by n
    new_array = np.arange(next_multiple_min, next_multiple_max + n, n)

    return new_array

# Define the function to convert eV to nm
def ev_to_nm(x):
    return 1240 / x

def nm_to_ev(x):
    return 1240 / x

##### PARAMETERS #####
# model_parent_path can also be given by "-m" parser argument, then the line below is not needed
# model_parent_path = '/data/cschmidt/excited_states/energy_NNs/fr0_project/trained_all_10' # path in which the best_model ist
data_folder_name = 'nn_data'
best_model_name = 'best_model' # name of the model
parameter_file_name = 'params.txt' # name of the file in which the parameters are stored
data_prefix = 'nn_input_'
log_file = 'summary.txt'

keep_energy_and_osc = True # Store Energy and Osc. Str. in separate files

# If you want to use the same size for all solvents, set this to True
UNIFORM_SIZE = False


# Shift for the histogram in nm
nm_from_which_to_shift_from = 368 # value from the Gaussian fit
nm_to_which_to_shift_to = 570 # value from the experiment

###################### END OF PARAMETERS ######################


###################### MAIN CODE ######################

parent_path = os.getcwd() # path for evaluation
data_folder_path = join(parent_path, data_folder_name)

data_names = [filename for filename in os.listdir(data_folder_path) if isfile(join(data_folder_path, filename))]
# all_solvent_names = ['acn', 'meoh', 'dcm', 'dmso', 'acetone', 'water', 'dmf', 'ethylacetate', 'hexane', 'dioxane']
all_solvent_names = [filename.split('_')[-1].split('.')[0] for filename in data_names]
# Sort the solvent names and data names by alphabet
sorted_solvent_data = sorted(zip(all_solvent_names, data_names), key=lambda x: x[0])
all_solvent_names, data_names = zip(*sorted_solvent_data)

data_paths = [join(data_folder_path, data_name) for data_name in data_names if os.path.exists(join(data_folder_path, data_name))]

natoms = extract_number_of_atoms(data_paths[0])

model_parent_path = args.model
model_path = join(model_parent_path, best_model_name)
model_name = model_parent_path.split('/')[-1]

model = tf.saved_model.load(model_path)

params = np.loadtxt(join(model_path, parameter_file_name), usecols=(0), max_rows=6)
#model.summary()
xmean, xstd, ymean, ystd, oscmean, oscstd = params[0:6]

A2Bohr = 1.8897259886
EhtoeV = 27.2114
ehtonm = 299792458 * 6.62607015e-7 / 4.3597482
eVtonm = 1.239841e3

# Shifts for the histogram in nm
shift = nm_to_ev(nm_to_which_to_shift_to) - nm_to_ev(nm_from_which_to_shift_from)

# Path for plots
plot_folder = join(parent_path, model_name)
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
else:
    shutil.rmtree(plot_folder)
    os.makedirs(plot_folder)

# Path for log file
log_file_path = join(plot_folder, log_file)
if os.path.exists(log_file_path):
    os.remove(log_file_path)

# Prepare logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler to the logger
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Determine size of the data used for all solvents, by searching for the smallest file
if UNIFORM_SIZE:
    size = None
    for testdata in data_paths:
        linestotal = get_file_length(testdata)
        ntotal = linestotal / (natoms + 1)
        if not ntotal.is_integer():
            raise ValueError("Number of Lines incorrect.")
        data_size = int(ntotal)
        if size is None:
            size = data_size
        else:
            size = min(size, data_size)

# Initialize a figure for the combined histogram
figure_combined, ax_combined = plt.subplots(figsize=(15, 5))
figure_spectrum, ax_spectrum = plt.subplots(figsize=(15, 5))
figure_spectrum_shift, ax_spectrum_shift = plt.subplots(figsize=(15, 5))
figure_spectrum_nm, ax_spectrum_nm = plt.subplots(figsize=(15, 5))

# Define plot colors
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
# colors_hist = ['lightskyblue', 'lightgreen', 'lightcoral', 'lightcyan', 'lightpink', 'lightyellow', 'lightgrey']
colors_hist = ['lightskyblue', 'lightgreen', 'lightcoral', 'lightcyan', 'lightpink', 'khaki', 'lightgrey', 'lightgoldenrodyellow', 'plum', 'burlywood']

logger.info(f"Starting evaluation of model {model_name} on all solvents from folder {data_folder_path}\n")

pred_eV_red_list, osc_str_red_list = [], []

# Loop over each dataset
for idx, (testdata, solvent_name) in enumerate(zip(data_paths, all_solvent_names)):
    # Get number of molecules
    result = subprocess.run(['wc', '-l', testdata], capture_output=True, text=True)
    max_n_molecules = int(result.stdout.split()[0])
    max_n_molecules /= natoms + 1

    if not UNIFORM_SIZE:
        linestotal = get_file_length(testdata)
        ntotal = linestotal / (natoms + 1)
        if not ntotal.is_integer():
            raise ValueError("Number of Lines incorrect.")
        size = int(ntotal)

    # Data extraction
    logger.info(f"{solvent_name}:")
    # print(f"Reading data from {testdata}")
    xyz_data = np.zeros((size, natoms, 4))
    # Read xyz data and ESP
    with open(testdata, "r") as data:
        for i in range(size):
            if i > max_n_molecules:
                raise ValueError("Something went horribly wrong. Look in the Code!")  # if size of data is smaller than the "size" variable
            for j in range(natoms):
                line = data.readline()
                xyz_data[i, j] = [float(x) for x in line.split()[1:]]
            data.readline()
    
    # Read atom types
    atom_types = []
    with open(testdata, "r") as data:
        for j in range(natoms):
            line = data.readline()
            atom_type = line.split()[0]
            atom_types.append(atom_type)
    # print("loaded")
    # logger.info(f"Data loaded from {testdata}")

    # Split data and adjust units
    xyz_data = np.asarray(xyz_data, dtype=np.float32)
    coords = xyz_data[:, :, :3]  # is (nrdata, 85, 3) or (nr, 170,3)
    coords *= A2Bohr
    esp_raw = xyz_data[:, :, 3]  # is (nrdata, 85)

    # Store test data in dictionary
    datas = {"x": coords, "esp": esp_raw}
    datas["x_scaled"] = (datas["x"] - xmean) / xstd

    # Prediction
    pred = model([datas["x_scaled"], datas["esp"]])
    predlist = pred.numpy()
    predlist[:, 0] = predlist[:, 0] * ystd + ymean
    predlist[:, 1] = predlist[:, 1] * oscstd + oscmean
    pred_eV = EhtoeV * predlist[:, 0]

    osc_str = predlist[:, 1]

    # Remove energies below a certain energy
    problematic_energy_threshold = 0
    osc_str_red = osc_str[pred_eV > problematic_energy_threshold]
    pred_eV_red = pred_eV[pred_eV > problematic_energy_threshold]

    # The energies and osc. str. for later use: replace problematic prediction with NANs
    pred_eV_copy = np.copy(pred_eV)
    osc_str_copy = np.copy(osc_str)

    pred_eV_copy[pred_eV < problematic_energy_threshold] = np.nan
    osc_str_copy[pred_eV < problematic_energy_threshold] = np.nan

    # Store for later use (storing in separate files)
    pred_eV_red_list.append(pred_eV_copy), osc_str_red_list.append(osc_str_copy)

    # Filter out weird anomalies
    pred_problems = pred_eV[pred_eV < problematic_energy_threshold]
    coords_problems = coords[pred_eV < problematic_energy_threshold]
    osc_str_problems = osc_str[pred_eV < problematic_energy_threshold]
    # print(f"Found {len(pred_problems)} problematic predictions in {solvent_name}")
    logger.info(f"Found {len(pred_problems)} problematic predictions in {solvent_name}")
    # Store the problematic predictions in one xyz file
    if len(pred_problems) > 0:
        # print(f"Storing problematic predictions in {solvent_name}_problems.xyz")
        logger.info("Storing problematic predictions")
        # write(f"{solvent_name}_problems.xyz", [Atoms(atom_types, positions=coords_problems[i]) for i in range(len(coords_problems))], comment=f"Energy={pred_problems}, Osc. Str={osc_str_problems}")
        problematic_file_path = join(plot_folder, f"{solvent_name}_problems.xyz")
        with open(problematic_file_path, 'w') as file:
            for i in range(len(coords_problems)):
                atoms = Atoms(atom_types, positions=coords_problems[i])
                write(problematic_file_path, atoms, append=True, comment=f"Energy={pred_problems[i]}, Osc. Str={osc_str_problems[i]}")

    # Color
    color = colors[idx]
    color_hist = colors_hist[idx]

    # Create individual histogram with normalization
    plt.figure()
    n, bins, patches = plt.hist(pred_eV_red, bins=100, weights=osc_str_red, density=True, color=color_hist, label=f'{solvent_name}')
    binwidth = bins[1] - bins[0]
    binmids = bins[:-1] + 0.5 * binwidth

    # Fit Gaussian
    popt, pcov = curve_fit(gaussian, binmids, n, p0=[3.5, 0.5])
    plt.plot(binmids, gaussian(binmids, *popt), color=color)
    # print(f"Mean: {popt[0]}, Variance: {popt[1]}")
    logger.info(f"{solvent_name} - Mean: {popt[0]}, Variance: {popt[1]}\n")

    plt.xlabel("Excitation Energy [eV]")
    plt.ylabel("Count [a.u.]")
    plt.legend()

    fig_name = f"histogram_{solvent_name}.png"
    fig_path = join(plot_folder, fig_name)
    plt.savefig(fig_path)
    plt.close()  # Close the individual plot to avoid overwriting

    # For all plots
    binmids_shifted = binmids + shift

    # Add to combined plot
    plt.figure(figure_combined.number)  # Setze figure_combined als aktive Figure
    # combined_n, combined_bins, combined_patches = plt.hist(pred_eV_red + shift, bins=100, weights=osc_str_red, density=True, alpha=0.5, color=color_hist, label=f'{solvent_name}')
    # plt.plot(binmids_shifted, gaussian(binmids, *popt), color=color, label=f'{solvent_name} fit')
    combined_n, combined_bins, combined_patches = plt.hist(pred_eV_red, bins=100, weights=osc_str_red, density=True, alpha=0.5, color=color_hist, label=f'{solvent_name}')
    plt.plot(binmids, gaussian(binmids, *popt), color=color, label=f'{solvent_name} fit')

    # Add to combined Gaussian plot
    plt.figure(figure_spectrum.number)  # Setze figure_spectrum als aktive Figure
    plt.plot(binmids, gaussian_A1(binmids, *popt), color=color, label=f'{solvent_name} {round(popt[0],2)} eV, {int(ev_to_nm(popt[0]))} nm')
    # plt.vlines(x=popt[0], ymin=0, ymax=gaussian_A1(popt[0], *popt), color=color, linestyle='--') #  vertical lines with max
    plt.vlines(x=popt[0], ymin=0, ymax=2, color=color, linestyle='--') # vertical lines with no max

    # Add to combined Gaussian plot
    plt.figure(figure_spectrum_shift.number)  # Setze figure_spectrum als aktive Figure
    plt.plot(binmids_shifted, gaussian_A1(binmids, *popt), color=color, label=f'{solvent_name} {round(popt[0] + shift,2)} eV, {int(ev_to_nm(popt[0] + shift))} nm')
    # plt.vlines(x=popt[0], ymin=0, ymax=gaussian_A1(popt[0], *popt), color=color, linestyle='--') #  vertical lines with max
    plt.vlines(x=popt[0] + shift, ymin=0, ymax=2, color=color, linestyle='--') # vertical lines with no max

    # Add to combined Gaussian plot in nm
    plt.figure(figure_spectrum_nm.number)  # Setze figure_spectrum_nm als aktive Figure
    plt.plot(eVtonm / binmids_shifted, gaussian_A1(binmids, *popt), color=color, label=f'{solvent_name} {round(popt[0],2)} eV, {int(ev_to_nm(popt[0]))} nm')
    plt.vlines(x=eVtonm / (popt[0] + shift), ymin=0, ymax=gaussian_A1(popt[0], *popt), color=color, linestyle='--')


# Parameters for all plots
xlim_bottom = 2.5
xlim_top = 4.0
nm_tick_distance = 10
n_th_nm_tick = 5

# Shifted values
xlim_bottom_shifted = 2.5 + shift
xlim_top_shifted = 4.0 + shift

##### PLOT1: Finalize the combined plot
plt.figure(figure_combined.number)  # Setze fig_combined als aktive Figure
ax_combined.set_xlabel("Excitation Energy [eV]")
ax_combined.set_ylabel("Count [a.u.]")
ax_combined.set_xlim(xlim_bottom, xlim_top)
ax_combined.set_ylim(0, 4)
handles, labels = ax_combined.get_legend_handles_labels()
figure_combined.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
fig_name = "histogram_all_solvents_combined.png"
fig_path = join(plot_folder, fig_name)
plt.savefig(fig_path, bbox_inches='tight')
plt.show()


##### PLOT2: Finalize the spectrum plot with secondary x-axis
plt.figure(figure_spectrum.number)  # Setze fig_spectrum als aktive Figure

# Bottom x-axis (Excitation Energy in eV)
ax_spectrum.set_xlabel("Excitation Energy [eV]")
ax_spectrum.set_ylabel("Count [a.u.]")
ax_spectrum.set_xlim(xlim_bottom, xlim_top)
ax_spectrum.set_ylim(0, 1.05)

# Create the secondary x-axis at the top (Wavelength in nm)
ax2 = ax_spectrum.twiny()  # Create the second x-axis (ax2 is for nm)

# Set the limits for the secondary x-axis (nm)
ax2.set_xlim(xlim_bottom, xlim_top)

# Customize top tick locations and labels (in nm, equally spaced)
# top_ticks_nm = np.linspace(ev_to_nm(xlim_bottom), ev_to_nm(xlim_top), num=8)  # Equally spaced ticks in nm
top_ticks_nm = multiples_of_n_in_between_interval(ev_to_nm(xlim_bottom), ev_to_nm(xlim_top), nm_tick_distance)  # Round to multiples of 50
# top_ticks_eV = nm_to_ev(top_ticks_nm)  # Convert nm ticks back to eV

# Customize top tick locations and labels (in nm, equally spaced)
top_ticks_nm_major = top_ticks_nm[::n_th_nm_tick]  # Every 5th tick for major ticks
top_ticks_eV_major = nm_to_ev(top_ticks_nm_major)  # Convert major nm ticks back to eV
top_ticks_eV_minor = nm_to_ev(top_ticks_nm)  # Convert all nm ticks back to eV for minor ticks

# Set major ticks on the top axis using the nm ticks converted back to eV
ax2.set_xticks(top_ticks_eV_major)  # Set the major eV ticks on the top axis
ax2.set_xticklabels([f'{int(tick)}' for tick in top_ticks_nm_major])  # Label with nm values for major ticks

# Set minor ticks on the top axis
ax2.set_xticks(top_ticks_eV_minor, minor=True)  # Set the minor eV ticks on the top axis

# Customize the appearance of the minor ticks
ax2.tick_params(axis='x', which='minor', length=4)  # Adjust the length of minor ticks
ax2.tick_params(axis='x', which='major', length=8)  # Adjust the length of major ticks

# # Set ticks on the top axis using the nm ticks converted back to eV
# ax2.set_xticks(top_ticks_eV)  # Set the nm ticks on the top axis
# ax2.set_xticklabels([f'{int(tick)}' for tick in top_ticks_nm])  # Label with nm values

# Add label to the secondary x-axis
ax2.set_xlabel("Wavelength [nm]")

# Create a single legend for all plot elements
handles, labels = ax_spectrum.get_legend_handles_labels()
figure_spectrum.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

# Save the figure
fig_name = "histogram_spectrum.png"
fig_path = join(plot_folder, fig_name)
plt.savefig(fig_path, bbox_inches='tight')
plt.show()


##### PLOT3: Finalize the spectrum plot with secondary x-axis
plt.figure(figure_spectrum_shift.number)  # Setze fig_spectrum als aktive Figure

# Bottom x-axis (Excitation Energy in eV)
ax_spectrum_shift.set_xlabel("Excitation Energy [eV]")
ax_spectrum_shift.set_ylabel("Count [a.u.]")
ax_spectrum_shift.set_xlim(xlim_bottom_shifted, xlim_top_shifted)
ax_spectrum_shift.set_ylim(0, 1.05)

# Create the secondary x-axis at the top (Wavelength in nm)
ax2 = ax_spectrum_shift.twiny()  # Create the second x-axis (ax2 is for nm)

# Set the limits for the secondary x-axis (nm)
ax2.set_xlim(xlim_bottom_shifted, xlim_top_shifted)

# Customize top tick locations and labels (in nm, equally spaced)
# top_ticks_nm = np.linspace(ev_to_nm(xlim_bottom_shifted), ev_to_nm(xlim_top_shifted), num=8)  # Equally spaced ticks in nm
top_ticks_nm = multiples_of_n_in_between_interval(ev_to_nm(xlim_bottom_shifted), ev_to_nm(xlim_top_shifted), nm_tick_distance)  # Round to multiples of 50
# top_ticks_eV = nm_to_ev(top_ticks_nm)  # Convert nm ticks back to eV

top_ticks_nm_major = top_ticks_nm[::n_th_nm_tick]  # Every 5th tick for major ticks
top_ticks_eV_major = nm_to_ev(top_ticks_nm_major)  # Convert major nm ticks back to eV
top_ticks_eV_minor = nm_to_ev(top_ticks_nm)  # Convert all nm ticks back to eV for minor ticks

# Set major ticks on the top axis using the nm ticks converted back to eV
ax2.set_xticks(top_ticks_eV_major)  # Set the major eV ticks on the top axis
ax2.set_xticklabels([f'{int(tick)}' for tick in top_ticks_nm_major])  # Label with nm values for major ticks

# Set minor ticks on the top axis
ax2.set_xticks(top_ticks_eV_minor, minor=True)  # Set the minor eV ticks on the top axis

# Customize the appearance of the minor ticks
ax2.tick_params(axis='x', which='minor', length=4)  # Adjust the length of minor ticks
ax2.tick_params(axis='x', which='major', length=8)  # Adjust the length of major ticks

# # Set ticks on the top axis using the nm ticks converted back to eV
# ax2.set_xticks(top_ticks_eV)  # Set the nm ticks on the top axis
# ax2.set_xticklabels([f'{int(tick)}' for tick in top_ticks_nm])  # Label with nm values

# Add label to the secondary x-axis
ax2.set_xlabel("Wavelength [nm]")

# Create a single legend for all plot elements
handles, labels = ax_spectrum_shift.get_legend_handles_labels()
figure_spectrum_shift.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

# Save the figure
fig_name = "histogram_spectrum_shift.png"
fig_path = join(plot_folder, fig_name)
plt.savefig(fig_path, bbox_inches='tight')
plt.show()


##### PLOT4: Finalize the spectrum plot in nanometers
plt.figure(figure_spectrum_nm.number)  # Setze fig_spectrum_nm als aktive Figure
ax_spectrum_nm.set_xlabel("Wavelength [nm]")
ax_spectrum_nm.set_ylabel("Count [a.u.]")
ax_spectrum_nm.set_xlim(ev_to_nm(xlim_top_shifted), ev_to_nm(xlim_bottom_shifted))
ax_spectrum_nm.set_ylim(0, 1.05)

ticks_nm = multiples_of_n_in_between_interval(ev_to_nm(xlim_bottom_shifted), ev_to_nm(xlim_top_shifted), nm_tick_distance)
ticks_nm_major = ticks_nm[::n_th_nm_tick]  # Every 5th tick for major ticks
ticks_nm_minor = ticks_nm

# Set major ticks on the top axis using the nm ticks converted back to eV
ax_spectrum_nm.set_xticks(ticks_nm_major)  # Set the major eV ticks on the top axis
ax_spectrum_nm.set_xticklabels([f'{int(tick)}' for tick in ticks_nm_major])  # Label with nm values for major ticks

# Set minor ticks on the top axis
ax_spectrum_nm.set_xticks(ticks_nm_minor, minor=True)  # Set the minor eV ticks on the top axis

# Customize the appearance of the minor ticks
ax_spectrum_nm.tick_params(axis='x', which='minor', length=4)  # Adjust the length of minor ticks
ax_spectrum_nm.tick_params(axis='x', which='major', length=8)  # Adjust the length of major ticks

handles, labels = ax_spectrum_nm.get_legend_handles_labels()
figure_spectrum_nm.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
fig_name = "histogram_spectrum_nm.png"
fig_path = join(plot_folder, fig_name)
plt.savefig(fig_path, bbox_inches='tight')
plt.show()

# Store energy and osc. str. data in separate files
if keep_energy_and_osc:
    for solvent_name, pred_eV_red, osc_str_red in zip(all_solvent_names, pred_eV_red_list, osc_str_red_list):
        file_name = f"{solvent_name}_energy_osc_str.txt"
        file_path = join(plot_folder, file_name)
        data = np.column_stack((pred_eV_red, osc_str_red))
        np.savetxt(file_path, data, header="Energy (eV) Oscillator Strength", comments='')