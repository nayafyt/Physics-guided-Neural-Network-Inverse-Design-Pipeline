"""
Result visualization and export utilities.
Handles CSV saving and plot generation for physics problems.
"""
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_predictions_csv(predictions, δobs, output_dir, Evals_int):
    """
    Export indentation predictions to CSV file.
    
    Args:
        predictions: [N, output_dim] physics model outputs
        δobs: [N, 1] indentation tensor
        output_dir: folder to save CSV
        Evals_int: [3] rounded predicted moduli for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    deltaex_np = δobs.squeeze(1).cpu().numpy()
    Fpred_np = predictions.squeeze(1).cpu().numpy()
    jobName = os.path.join(output_dir, f"Pred_{Evals_int[0]}_{Evals_int[1]}_{Evals_int[2]}.csv")
    
    with open(jobName, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['deltaex', 'Fpred'])
        for d, fp in zip(deltaex_np, Fpred_np):
            writer.writerow([d, fp])
    print(f"Saved predictions to {jobName}")

def save_vessel_design_csv(predictions, physics_output, output_dir, design_int):
    """
    Export vessel design optimization results to CSV file.
    Mirrors the indentation format: design variables + physics output (min_val).
    
    Args:
        predictions: [batch_size, 5] continuous design predictions from PGNN
                     [SAngle, Nrplies, Stepply, SymLam, Thickpl]
        physics_output: [batch_size, 1] corresponding min_val objectives from forward_physics
        output_dir: folder to save CSV
        design_int: [5] rounded predicted design variables for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pred_np = predictions.cpu().numpy() if predictions.dim() > 1 else predictions.cpu().numpy()
    phys_np = physics_output.squeeze(1).cpu().numpy() if physics_output.dim() > 1 else physics_output.cpu().numpy()
    
    jobName = os.path.join(output_dir, f"optimal_design.csv")
    
    with open(jobName, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SAngle', 'Nrplies', 'Stepply','SymLam', 'Thickpl', 'min_val'])
        for pred, mv in zip(pred_np, phys_np):
            writer.writerow([pred[0], pred[1], pred[2], pred[3], pred[4], mv])
    print(f"Saved vessel design to {jobName}")

def save_vessel_epoch_results_csv(epoch_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    jobName = os.path.join(output_dir, "epoch_results.csv")

    obj_dim = epoch_results[0]['physics_output'].shape[-1]

    with open(jobName, 'w', newline='') as f:
        writer = csv.writer(f)
        design_cols = ['Epoch', 'SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl']
        if obj_dim == 1:
            writer.writerow(design_cols + ['min_val'])
        else:
            writer.writerow(design_cols + ['obj_S11', 'obj_S22', 'obj_Thick'])
        for result in epoch_results:
            pred = result['predictions'][0]
            phys = result['physics_output'][0]  # [K]
            writer.writerow([result['epoch'], *pred, *phys])
    print(f"Saved epoch results to {jobName}")

def plot_force_indentation(δobs, Fobs, Fpred_full, output_dir):
    """
    Plot measured vs predicted force-indentation curve.
    
    Args:
        δobs: [N, 1] indentation tensor
        Fobs: [N, 1] observed force tensor
        Fpred_full: [N, 1] predicted force tensor
        output_dir: folder to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    print("δobs shape:", δobs.shape)
    print("Fobs shape:", Fobs.shape)
    print("Fpred_full shape:", Fpred_full.shape)

    plt.figure()
    plt.plot(δobs, Fobs, 'bo', label='measured')
    plt.plot(δobs, Fpred_full[:, 0], 'r-', label='predicted')
    plt.plot(δobs, Fpred_full[:, 1:], 'r-', alpha=0.3)
    plt.xlabel("Indentation δ (m)")
    plt.ylabel("Force F (N)")
    plt.title("Measured vs Predicted Force–Indentation Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "force_indentation.png"), dpi=300)
    plt.close()

def plot_loss_curves(history, output_dir):
    """
    Plot training loss curves.
    
    Args:
        history: dict with 'total', 'data', 'constraint' loss lists
        output_dir: folder to save plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(history['total'], label='total')
    plt.plot(history['data'], label='data')
    plt.plot(history['constraint'], label='constraint')
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=300)
    plt.close()

def evaluate_rank(epoch_results_path, dataset_path, objective_mode='min_val'):
    epoch_df = pd.read_csv(epoch_results_path)
    vessel_df = pd.read_csv(dataset_path)
    vessel_df.columns = vessel_df.columns.str.strip()

    design_cols = ['SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl']

    last_row = epoch_df.iloc[-1]
    predicted = np.array([last_row[c] for c in design_cols])

    print("\nPredicted design from last epoch:")
    print(predicted)

    X = vessel_df[design_cols].values
    dists = np.linalg.norm(X - predicted, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_row = vessel_df.iloc[nearest_idx]

    s11_norm = nearest_row['S11'] / 2500.0
    s22_norm = nearest_row['S22'] / 185.0
    thick_norm = nearest_row['Thick'] * 0.12
    min_val = nearest_row['min_val']

    print(f"\nNearest dataset design:")
    print(X[nearest_idx])
    print(f"\n  S11/2500:    {s11_norm:.4f}  (S11 = {nearest_row['S11']:.1f})")
    print(f"  S22/185:     {s22_norm:.4f}  (S22 = {nearest_row['S22']:.1f})")
    print(f"  Thick*0.12:  {thick_norm:.4f}  (Thick = {nearest_row['Thick']:.3f})")
    print(f"  min_val:     {min_val:.6f}")

    # Primary rank is always by min_val
    all_min_val = vessel_df["min_val"].values
    rank = (np.argsort(np.argsort(all_min_val))[nearest_idx]) + 1
    print(f"\nRank in dataset (by min_val): {rank} / {len(vessel_df)}")
    print(f"  Nearest: {min_val:.6f}  |  Dataset best: {all_min_val.min():.6f}")

    if objective_mode == 'multi_objective':
        obj_names = ['S11/2500', 'S22/185', 'Thick*0.12']
        all_s11 = vessel_df["S11"].values / 2500.0
        all_s22 = vessel_df["S22"].values / 185.0
        all_thick = vessel_df["Thick"].values * 0.12
        all_objs = [all_s11, all_s22, all_thick]
        nearest_vals = [s11_norm, s22_norm, thick_norm]

        print(f"\nSupporting objective ranks:")
        for name, col, val in zip(obj_names, all_objs, nearest_vals):
            rank_i = (np.argsort(np.argsort(col))[nearest_idx]) + 1
            print(f"  {name:12s}: {val:.4f}  |  rank {rank_i:5d} / {len(vessel_df)}  (best: {col.min():.4f})")

    print(f"\nDistance to nearest design: {dists[nearest_idx]:.6f}")
    print("====================================\n")


def plot_multi_objective_curves(epoch_results_path, dataset_path, output_dir):
    epoch_df = pd.read_csv(epoch_results_path)
    vessel_df = pd.read_csv(dataset_path)
    vessel_df.columns = vessel_df.columns.str.strip()

    design_cols = ['SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl']
    X = vessel_df[design_cols].values

    epochs, s11_vals, s22_vals, thick_vals = [], [], [], []

    for _, row in epoch_df.iterrows():
        pred = np.array([row[c] for c in design_cols])
        dists = np.linalg.norm(X - pred, axis=1)
        nearest = vessel_df.iloc[np.argmin(dists)]

        epochs.append(row['Epoch'])
        s11_vals.append(nearest['S11'] / 2500.0)
        s22_vals.append(nearest['S22'] / 185.0)
        thick_vals.append(nearest['Thick'] * 0.12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, s11_vals, 'b-o', markersize=2)
    axes[0].set_title('S11 / 2500')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Value')

    axes[1].plot(epochs, s22_vals, 'r-o', markersize=2)
    axes[1].set_title('S22 / 185')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value')

    axes[2].plot(epochs, thick_vals, 'g-o', markersize=2)
    axes[2].set_title('Thick * 0.12')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value')

    fig.suptitle('Multi-Objective Components Over Training')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "multi_objective_curves.png"), dpi=300)
    plt.close(fig)
    print(f"Saved multi-objective curves to {output_dir}/multi_objective_curves.png")
