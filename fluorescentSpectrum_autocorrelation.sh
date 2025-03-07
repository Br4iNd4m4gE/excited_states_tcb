#!/home/cschmidt/miniconda3/envs/excited_states_NN_kgcnn/bin/python

# This script uses the following paper as reference:
# J. Chem. Phys. 151, 074111 (2019)
# https://doi.org/10.1063/1.5114818

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, isdir, isfile, dirname, abspath

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("file", type=str, help="Path to the input file")
args = ap.parse_args()
input_file = args.file

# function for loading excitation energies
def load_excitation_energies():
    return energies_over_time

def sigma_linear_absorption_spectrum(omegas, chi, times, dt, alpha_omega=1):
    """
    Equation 2
    performs fourier transformation of chi over time
    """
    sigma = alpha_omega * np.sum(chi * np.exp(-1j * omegas * times) * dt)
    spectrum = np.real(sigma)
    return spectrum

def chi_cumulant_approx_response_function(G_m, times, omega_av_eg, V_eg=1):
    """
    Equation 21
    note: G_m and times are arrays
    """
    chi = np.abs(V_eg) ** 2 * np.exp(-1j * omega_av_eg * times - G_m)
    return chi

def g_2(J, omega, time, beta):
    """
    Equation 22
    Integrate J / omega^2 by omega using the trapezoidal rule
    """
    integrand = J / (omega ** 2) * (np.cot(beta * omega / 2) * (1 - np.cos(omega * time)) - 1j * (np.sin(omega * time) - omega * time))
    integral = np.trapz(integrand, omega)
    g_2 = 1 / np.pi * integral
    return g_2

energies_over_time = load_excitation_energies(input_file) # energies given in eV

hbar = 6.582e-16  # Planck's constant in eV·s
hbar_fs = 6.582e-1  # Planck's constant in eV·fs
simulation_time_step = 0.5 # fs

times = np.linspace(0, simulation_time_step * len(energies_over_time)-1, len(energies_over_time))
omegas = energies_over_time / hbar_fs


# Calculate: C_deltaU - the energy gap fluctuation equilibrium time correlation function



# Calculate: J(omega)
J_omega = 1j * np.sum(np.exp(1j * omega * times) * np.imag(C_deltaU))


# Calculate: g_m's
g_2 = g_2(J, omegas, times, beta)

# Calculate: G_m
G_m = np.sum(g_ms)

# Calculate: omega_av_eg <- thermally averaged energy gap operator; average over U(R) vertical excitation energy (after Equation 17)
omega_av_eg = np.mean(omegas) # ? is it omegas or energies_over_time?

# Step X-1: Calculate the mth order cumulant approximation to the response function
chi_m = chi_cumulant_approx_response_function(G_m, times, omega_av_eg, V_eg=1)

# Last Step: Calculate the absorption spectrum
spectrum = sigma_linear_absorption_spectrum(omegas, chi_m, times, simulation_time_step, alpha_omega=1)