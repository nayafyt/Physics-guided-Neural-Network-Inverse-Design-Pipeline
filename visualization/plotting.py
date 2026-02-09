"""
Result visualization and export utilities.
Handles CSV saving and plot generation for physics problems.
"""
import os
import csv
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
                     [SAngle, Stepply, Nrplies, SymLam, Thickpl]
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
        writer.writerow(['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl', 'min_val'])
        for pred, mv in zip(pred_np, phys_np):
            writer.writerow([pred[0], pred[1], pred[2], pred[3], pred[4], mv])
    print(f"Saved vessel design to {jobName}")

def save_vessel_epoch_results_csv(epoch_results, output_dir):
    """
    Export vessel design results for each epoch to CSV file.
    Shows how the design evolved throughout training.
    
    Args:
        epoch_results: List of dicts, each containing:
                      {'epoch': int, 'predictions': [5], 'physics_output': [1]}
        output_dir: folder to save CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    jobName = os.path.join(output_dir, "epoch_results.csv")
    
    with open(jobName, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl', 'min_val'])
        for result in epoch_results:
            epoch = result['epoch']
            pred = result['predictions'][0]  # batch_size=1, get first (only) element
            phys = result['physics_output'][0, 0]  # batch_size=1, get scalar
            writer.writerow([epoch, pred[0], pred[1], pred[2], pred[3], pred[4], phys])
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
