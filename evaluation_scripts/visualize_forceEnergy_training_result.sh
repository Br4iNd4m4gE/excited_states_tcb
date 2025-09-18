#!/home/cschmidt/miniconda3/envs/dftb_workflow/bin/python
# filepath: /home/cschmidt/bin_public/excited_states_networks/evaluation_scripts/visualize_forceEnergy_training_result.sh

# Script to create heatmaps for energy and force predictions vs reference values
# Run this script in a directory containing:
# - energy_predictions.txt and energy_ref.txt (for energy heatmap)
# - force_predictions.txt and force_ref.txt (for force heatmap)
# Script will create heatmaps only for available data pairs

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, r2_score

threshold = 100  # Minimum count for force heatmap

# Conversion factors
EV_TO_KCAL_MOL = 23.06052  # 1 eV = 23.06052 kcal/mol

def check_file_availability():
    """Check which files are available and return availability status"""
    energy_files = ['energy_predictions.txt', 'energy_ref.txt']
    force_files = ['force_predictions.txt', 'force_ref.txt']
    
    energy_available = all(os.path.exists(f) for f in energy_files)
    force_available = all(os.path.exists(f) for f in force_files)
    
    if energy_available:
        print("✓ Energy files found: energy_predictions.txt, energy_ref.txt")
    else:
        missing_energy = [f for f in energy_files if not os.path.exists(f)]
        print(f"✗ Energy files missing: {missing_energy}")
    
    if force_available:
        print("✓ Force files found: force_predictions.txt, force_ref.txt")
    else:
        missing_force = [f for f in force_files if not os.path.exists(f)]
        print(f"✗ Force files missing: {missing_force}")
    
    if not energy_available and not force_available:
        print("Error: No complete data pairs found!")
        return False, False
    
    return energy_available, force_available

def load_energy_data():
    """Load energy prediction and reference data"""
    try:
        # Load energy data and convert eV to kcal/mol
        energy_pred = np.loadtxt('energy_predictions.txt') * EV_TO_KCAL_MOL
        energy_ref = np.loadtxt('energy_ref.txt') * EV_TO_KCAL_MOL
        
        # Convert to relative energies (subtract minimum)
        energy_min = min(energy_pred.min(), energy_ref.min())
        energy_pred_rel = energy_pred - energy_min
        energy_ref_rel = energy_ref - energy_min
        
        print(f"Loaded {len(energy_pred)} energy predictions (converted to relative energies)")
        print(f"Energy range: {energy_pred_rel.min():.1f} to {energy_pred_rel.max():.1f} kcal/mol")
        
        return energy_pred_rel, energy_ref_rel
    except Exception as e:
        print(f"Error loading energy data: {e}")
        return None, None

def load_force_data():
    """Load force prediction and reference data"""
    try:
        # Load force data and convert eV/A to kcal/mol/A
        force_pred = np.loadtxt('force_predictions.txt') * EV_TO_KCAL_MOL
        force_ref = np.loadtxt('force_ref.txt') * EV_TO_KCAL_MOL
        
        print(f"Loaded {len(force_pred)} force predictions")
        print(f"Force range: {force_pred.min():.1f} to {force_pred.max():.1f} kcal/mol/Å")
        
        return force_pred, force_ref
    except Exception as e:
        print(f"Error loading force data: {e}")
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

    prefix = ''

    if title == "Energy":
        # Custom formatting for energy: show "0" instead of "0.000"
        tick_labels = []
        for pos in tick_positions:
            if abs(pos) < 0.001:  # If very close to zero
                tick_labels.append("0")
            else:
                tick_labels.append(f"{pos:.1f}")  # 1 decimal place for others
        
        plt.xticks(tick_positions, labels=tick_labels)
        plt.yticks(tick_positions, labels=tick_labels)
        prefix = "(rel.)"
    else:
        # For forces: normal formatting
        plt.xticks(tick_positions)
        plt.yticks(tick_positions)

    # Labels
    plt.xlabel(f"Prediction {title} {prefix} [{unit}]", fontsize=FONTSIZE)
    plt.ylabel(f"Reference {title} {prefix} [{unit}]", fontsize=FONTSIZE)

    # Tick parameters
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    
    # Add error statistics (R² and MAE instead of RMSE)
    text_x = 0.05*(value_max-value_min) + value_min
    text_y = 0.85*(value_max-value_min) + value_min
    plt.text(text_x, text_y, f"MAE: {mae:.3f} {unit}\nR²: {r2:.3f}", 
             bbox={"facecolor": "white", "alpha": 0.8, "pad": 10}, fontsize=LABELSIZE)
    
    plt.tight_layout()
    plt.savefig(f"{save_name}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    threshold_info = f" (threshold: ≥{min_count})" if min_count > 1 else ""
    print(f"Created {save_name}_heatmap.png - MAE: {mae:.3f} {unit}, R²: {r2:.3f}{threshold_info}")

def main():
    """Main function to create heatmaps"""
    
    print("Checking file availability...")
    energy_available, force_available = check_file_availability()
    
    if not energy_available and not force_available:
        return
    
    created_plots = []
    
    # Process energy data if available
    if energy_available:
        print("\nProcessing energy data...")
        energy_pred, energy_ref = load_energy_data()
        if energy_pred is not None and energy_ref is not None:
            print("Creating energy heatmap...")
            create_heatmap(energy_pred, energy_ref, "Energy", r"$kcal\,mol^{-1}$", "energy")
            created_plots.append("energy_heatmap.png")
    
    # Process force data if available
    if force_available:
        print("\nProcessing force data...")
        force_pred, force_ref = load_force_data()
        if force_pred is not None and force_ref is not None:
            print("Creating force heatmap (normal)...")
            create_heatmap(force_pred, force_ref, "Forces", r"$kcal\,mol^{-1}\,\AA^{-1}$", "forces")
            created_plots.append("forces_heatmap.png")
            
            print("Creating force heatmap (with threshold)...")
            create_heatmap(force_pred, force_ref, "Forces", r"$kcal\,mol^{-1}\,\AA^{-1}$", "forces_threshold", min_count=threshold)
            created_plots.append("forces_threshold_heatmap.png")
    
    # Summary
    print(f"\nDone! Created {len(created_plots)} plot(s):")
    for plot in created_plots:
        print(f"  - {plot}")

if __name__ == "__main__":
    main()