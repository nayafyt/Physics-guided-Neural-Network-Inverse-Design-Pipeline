"""
Result visualization and export utilities.
Handles CSV saving and plot generation for physics problems.
"""
import os
import csv
import matplotlib.pyplot as plt

def save_predictions_csv(predictions, δobs, output_dir, Evals_int):
    """
    Export predictions to CSV file.
    
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
    plt.plot(δobs, Fobs, 'bo', label='meas')
    plt.plot(δobs, Fpred_full, 'r-', label='pred')
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
