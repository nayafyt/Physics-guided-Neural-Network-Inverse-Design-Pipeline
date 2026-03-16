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

    with open(jobName, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl', 'min_val'])
        for result in epoch_results:
            pred = result['predictions'][0]
            phys = result['physics_output'][0]  # [1]
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

def evaluate_rank(epoch_results_path, dataset_path):
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

    min_val = nearest_row['min_val']

    print(f"\nNearest dataset design:")
    print(X[nearest_idx])
    print(f"\n  S11/2500:    {nearest_row['S11'] / 2500.0:.4f}  (S11 = {nearest_row['S11']:.1f})")
    print(f"  S22/185:     {nearest_row['S22'] / 185.0:.4f}  (S22 = {nearest_row['S22']:.1f})")
    print(f"  Thick*0.12:  {nearest_row['Thick'] * 0.12:.4f}  (Thick = {nearest_row['Thick']:.3f})")
    print(f"  min_val:     {min_val:.6f}")

    all_min_val = vessel_df["min_val"].values
    rank = (np.argsort(np.argsort(all_min_val))[nearest_idx]) + 1
    print(f"\nRank in dataset (by min_val): {rank} / {len(vessel_df)}")
    print(f"  Nearest: {min_val:.6f}  |  Dataset best: {all_min_val.min():.6f}")

    print(f"\nDistance to nearest design: {dists[nearest_idx]:.6f}")
    print("====================================\n")
