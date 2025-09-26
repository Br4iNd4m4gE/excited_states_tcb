#!/home/cschmidt/miniconda3/envs/dftb_workflow/bin/python
# Script to create heatmaps for excitation energy and oscillator strength predictions vs reference values
# Run this script in a directory containing:
# - model_predicted_energies_eV.dat and model_ref_energies_eV.dat (for excitation energy heatmap)
# - model_predicted_osc.dat and model_ref_osc.dat (for oscillator strength heatmap)
# Script will create heatmaps only for available data pairs

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, r2_score

threshold = 100  # Minimum count for oscillator strength heatmap

def check_file_availability():
    """Check which files are available and return availability status"""
    energy_files = ['model_predicted_energies_eV.dat', 'model_ref_energies_eV.dat']
    osc_files = ['model_predicted_osc.dat', 'model_ref_osc.dat']
    
    energy_available = all(os.path.exists(f) for f in energy_files)
    osc_available = all(os.path.exists(f) for f in osc_files)
    
    if energy_available:
        print("✓ Excitation energy files found: model_predicted_energies_eV.dat, model_ref_energies_eV.dat")
    else:
        missing_energy = [f for f in energy_files if not os.path.exists(f)]
        print(f"✗ Excitation energy files missing: {missing_energy}")
    
    if osc_available:
        print("✓ Oscillator strength files found: model_predicted_osc.dat, model_ref_osc.dat")
    else:
        missing_osc = [f for f in osc_files if not os.path.exists(f)]
        print(f"✗ Oscillator strength files missing: {missing_osc}")
    
    if not energy_available and not osc_available:
        print("Error: No complete data pairs found!")
        return False, False
    
    return energy_available, osc_available

def load_excitation_energy_data():
    """Load excitation energy prediction and reference data"""
    try:
        # Load excitation energy data (already in eV according to filename)
        energy_pred = np.loadtxt('model_predicted_energies_eV.dat')
        energy_ref = np.loadtxt('model_ref_energies_eV.dat')
        
        print(f"Loaded {len(energy_pred)} excitation energy predictions")
        print(f"Excitation energy range: {energy_pred.min():.2f} to {energy_pred.max():.2f} eV")
        
        return energy_pred, energy_ref
    except Exception as e:
        print(f"Error loading excitation energy data: {e}")
        return None, None

def load_oscillator_strength_data():
    """Load oscillator strength prediction and reference data"""
    try:
        # Load oscillator strength data (unitless)
        osc_pred = np.loadtxt('model_predicted_osc.dat')
        osc_ref = np.loadtxt('model_ref_osc.dat')
        
        print(f"Loaded {len(osc_pred)} oscillator strength predictions")
        print(f"Oscillator strength range: {osc_pred.min():.3f} to {osc_pred.max():.3f}")
        
        return osc_pred, osc_ref
    except Exception as e:
        print(f"Error loading oscillator strength data: {e}")
        return None, None

def create_heatmap(predictions, references, title, unit, save_name, min_count=1):
    """Create heatmap plot matching data_plotter.py style"""
    
    plt.figure(figsize=(8, 6))
    FONTSIZE = 18
    LABELSIZE = 16
    
    # Calculate statistics
    mae = mean_absolute_error(references, predictions)
    r2 = r2_score(references, predictions)
    
    # Get the Blues colormap and truncate it to skip the very light/white parts
    blues_cmap = cm.get_cmap('Blues')
    # Use only the range from 0.2 to 1.0 (skip the white/very light blue part)
    truncated_blues = plt.matplotlib.colors.ListedColormap(blues_cmap(np.linspace(0.2, 1.0, 256)))
    
    # Create hexbin heatmap with threshold
    hb = plt.hexbin(predictions, references, gridsize=50, cmap=truncated_blues, mincnt=min_count)
    
    # Add colorbar with frequency label (and threshold info if > 1)
    cbar = plt.colorbar(hb)
    if min_count > 1:
        cbar.set_label(f'Frequency (≥{min_count})', fontsize=LABELSIZE)
    else:
        cbar.set_label('Frequency', fontsize=LABELSIZE)
    
    # Perfect prediction line (diagonal)
    value_min = min(predictions.min(), references.min())
    value_max = max(predictions.max(), references.max())
    plt.plot([value_min-0.1*(value_max-value_min), value_max+0.1*(value_max-value_min)], 
             [value_min-0.1*(value_max-value_min), value_max+0.1*(value_max-value_min)], 
             "k-", linewidth=2, label="Perfect Prediction")
    
    # Set equal axis limits and ticks
    plt.xlim(value_min, value_max)
    plt.ylim(value_min, value_max)
    
    # Make x and y axis ticks equal with custom formatting
    tick_positions = np.linspace(value_min, value_max, 6)  # 6 equally spaced ticks
    
    if title == "Excitation Energy":
        # Custom formatting for excitation energies: 1 decimal place
        tick_labels = [f"{pos:.1f}" for pos in tick_positions]
        plt.xticks(tick_positions, labels=tick_labels)
        plt.yticks(tick_positions, labels=tick_labels)
    else:
        # For oscillator strengths: 2 decimal places for better precision
        tick_labels = [f"{pos:.2f}" for pos in tick_positions]
        plt.xticks(tick_positions, labels=tick_labels)
        plt.yticks(tick_positions, labels=tick_labels)

    # Labels
    if unit:
        plt.xlabel(f"Prediction {title} [{unit}]", fontsize=FONTSIZE)
        plt.ylabel(f"Reference {title} [{unit}]", fontsize=FONTSIZE)
    else:
        plt.xlabel(f"Prediction {title}", fontsize=FONTSIZE)
        plt.ylabel(f"Reference {title}", fontsize=FONTSIZE)

    # Tick parameters
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    
    # Add error statistics (R² and MAE)
    text_x = 0.05*(value_max-value_min) + value_min
    text_y = 0.85*(value_max-value_min) + value_min
    
    if unit:
        mae_text = f"MAE: {mae:.3f} {unit}"
    else:
        mae_text = f"MAE: {mae:.3f}"
    
    plt.text(text_x, text_y, f"{mae_text}\nR²: {r2:.3f}", 
             bbox={"facecolor": "white", "alpha": 0.8, "pad": 10}, fontsize=LABELSIZE)
    
    plt.tight_layout()
    plt.savefig(f"{save_name}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    threshold_info = f" (threshold: ≥{min_count})" if min_count > 1 else ""
    unit_text = f" {unit}" if unit else ""
    print(f"Created {save_name}_heatmap.png - MAE: {mae:.3f}{unit_text}, R²: {r2:.3f}{threshold_info}")

def main():
    """Main function to create heatmaps"""
    
    print("Checking file availability...")
    energy_available, osc_available = check_file_availability()
    
    if not energy_available and not osc_available:
        return
    
    created_plots = []
    
    # Process excitation energy data if available
    if energy_available:
        print("\nProcessing excitation energy data...")
        energy_pred, energy_ref = load_excitation_energy_data()
        if energy_pred is not None and energy_ref is not None:
            print("Creating excitation energy heatmap...")
            create_heatmap(energy_pred, energy_ref, "Excitation Energy", "eV", "excitation_energy")
            created_plots.append("excitation_energy_heatmap.png")
    
    # Process oscillator strength data if available
    if osc_available:
        print("\nProcessing oscillator strength data...")
        osc_pred, osc_ref = load_oscillator_strength_data()
        if osc_pred is not None and osc_ref is not None:
            print("Creating oscillator strength heatmap (normal)...")
            create_heatmap(osc_pred, osc_ref, "Oscillator Strength", "", "oscillator_strength")
            created_plots.append("oscillator_strength_heatmap.png")
            
            print("Creating oscillator strength heatmap (with threshold)...")
            create_heatmap(osc_pred, osc_ref, "Oscillator Strength", "", "oscillator_strength_threshold", min_count=threshold)
            created_plots.append("oscillator_strength_threshold_heatmap.png")
    
    # Summary
    print(f"\nDone! Created {len(created_plots)} plot(s):")
    for plot in created_plots:
        print(f"  - {plot}")

if __name__ == "__main__":
    main()