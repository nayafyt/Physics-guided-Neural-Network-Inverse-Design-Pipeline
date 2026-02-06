"""
PGNN inverse hyperelastic training with:
  - Deterministic setup (fixed random seeds)
  - Cosine-annealed learning rate for exploration/exploitation
  - Hinge loss enforcing E2 >= E3 (ensures layer stiffness ordering)
  - Early stopping once the mean moduli stabilize at the first decimal
    (configurable number of consecutive stable epochs)
  - CSV export of final predicted force-indentation curve
    (filename includes predicted average moduli)
  - Timer for optimization duration
"""
import time
import random
import numpy as np
import torch  #Open-source machine learning library
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import math
from scipy.optimize import curve_fit
import os
from physics.indentation import IndentationProblem
from physics.vessel import VesselProblem

# Import visualization utilities
from visualization.plotting import save_predictions_csv, plot_force_indentation, plot_loss_curves

# Set output root directory from environment variable or default
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "results")

# 1) DETERMINISM -------------------------------------------------- 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# 2) MODEL DEFINITION ---------------------------------------------
class CustomActivation(nn.Module):
    """Maps raw network outputs to [E_min, E_max] via tanh scaling."""
    def __init__(self, E_min, E_max):
        super().__init__()
        self.E_min, self.E_max = E_min, E_max
    def forward(self, x):
        # tanh in [-1,1] -> scale to [E_min,E_max]
        return self.E_min + (self.E_max - self.E_min) * (torch.tanh(x) + 1) / 2

class PGNN(nn.Module):
    """
    Generic Physics-Guided NN with configurable input/output dims and bounds.
    - For indentation (forward): input_dim=2, output_dim=3
    - For inverse design: input_dim=0 → uses learnable latent parameters
    
    Args:
        input_dim (int): Input dimension (0 for inverse design)
        output_dim (int): Output dimension (number of parameters to predict)
        bounds (list): List of (min, max) tuples for each output
        num_hidden_layers (int): Number of hidden layers (default: 3)
        hidden_dim (int): Size of each hidden layer (default: 72)
    """
    def __init__(self, input_dim, output_dim, bounds, num_hidden_layers=3, hidden_dim=72):
        super().__init__()

        # For inverse design (input_dim=0), use learnable latent parameters
        if input_dim == 0:
            self.latent_params = nn.Parameter(torch.randn(1, 16) * 0.1)
            actual_input_dim = 16
        else:
            self.latent_params = None
            actual_input_dim = input_dim

        layers = []
        prev = actual_input_dim
        for _ in range(num_hidden_layers):  # Configurable number of hidden layers
            layers.append(nn.Linear(prev, hidden_dim))
            # Use LayerNorm for inverse design (batch_size=1), BatchNorm for forward (batch_size>1)
            if input_dim == 0:
                layers.append(nn.LayerNorm(hidden_dim))  # Inverse design with latent params
            else:
                layers.append(nn.BatchNorm1d(hidden_dim))  # Forward design with input data
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.hidden = nn.Sequential(*layers)

        self.activations = nn.ModuleList([
            CustomActivation(bounds[i][0], bounds[i][1])
            for i in range(output_dim)
        ])

    def forward(self, x):
        # If using latent parameters (inverse design), expand them
        if self.latent_params is not None:
            # Repeat latent params to match batch size
            latent = self.latent_params.expand(x.shape[0], -1)
            net_input = latent
        else:
            net_input = x
        
        raw = self.hidden(net_input)
        outputs = []
        for i in range(len(self.activations)):
            Ei = self.activations[i](raw[:, i].unsqueeze(1)).squeeze(1)
            outputs.append(Ei)
        return torch.stack(outputs, dim=1)

def init_weights(m):
    """Xavier init for Linear, standard for BN."""
    if isinstance(m, nn.Linear):
        gain = nn.init.calculate_gain('tanh')
        std = gain * math.sqrt(2.0/(m.weight.size(0)+m.weight.size(1)))
        nn.init.normal_(m.weight,0.,std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# 5) Generic loss & training-------------------------------------------

def loss_fn(model, inp, obs, problem, wOrder=1.0):
    """
    Generic loss:
        total = MSE(obs, physics_output) + wOrder * constraint_loss
    """
    predictions = model(inp)
    physics_output = problem.forward_physics(inp, predictions)
    data_loss = torch.mean((obs - physics_output)**2)
    constraint = problem.constraint_loss(predictions)
    total_loss = data_loss + wOrder * constraint
    return total_loss, data_loss, constraint, predictions, physics_output

def train(problem, bounds, data_path,
          num_hidden_layers=3, hidden_dim=72,
          patience=250, tighten_epochs=1500, stable_epochs=6):
    """
    Generic training loop for any PhysicsProblem.
    
    Args:
        problem: PhysicsProblem instance (Indentation or Vessel)
        bounds: List of (min, max) tuples for each output parameter
        data_path: Path to CSV file with training data
        num_hidden_layers: Number of hidden layers in PGNN (default: 3)
        hidden_dim: Size of each hidden layer (default: 72)
        patience: Epochs without improvement before early stopping (default: 250)
        tighten_epochs: Maximum training epochs (default: 1500)
        stable_epochs: Consecutive stable epochs before stopping (default: 6)
    """
    t0 = time.time()

    input_dim, output_dim = problem.get_input_output_dims()
    model = PGNN(input_dim, output_dim, bounds, num_hidden_layers=num_hidden_layers, hidden_dim=hidden_dim)
    model.apply(init_weights)
    
    # Optimize both hidden layer + latent params (if exists)
    if hasattr(model, 'latent_params') and model.latent_params is not None:
        optimizer = optim.Adam(list(model.hidden.parameters()) + [model.latent_params], lr=1e-3)
    else:
        optimizer = optim.Adam(model.hidden.parameters(), lr=1e-3)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=tighten_epochs, eta_min=1e-5)

    inp, obs = problem.load_data(data_path)
    history = {'total': [], 'data': [], 'constraint': []}
    best_loss = float('inf')
    no_imp = 0
    stable_count = 0
    prev_rounded = None

    for epoch in range(1, 10001):
        optimizer.zero_grad()
        total, dL, oL, pred, phys = loss_fn(model, inp, obs, problem, wOrder=1.0)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['total'].append(total.item())
        history['data'].append(dL.item())
        history['constraint'].append(oL.item())

        # check improvement
        if total.item() < best_loss:
            best_loss, no_imp = total.item(), 0
        else:
            no_imp += 1

        # first-decimal stability early-stop
        #On our run this got trigger first before no improvement condition
        pred_means = pred.mean(dim=0)
        rounded = torch.round(pred_means * 10) / 10 # first decimal
        if prev_rounded is not None and torch.all(rounded == prev_rounded):
            stable_count += 1
        else:
            stable_count = 0
        prev_rounded = rounded

        # stopping conditions
        if no_imp >= patience: #That (NO improvement condition) did not get trigger 
            print(f"Early stop (no loss imp.) @ epoch {epoch}")
            break
        if stable_count >= stable_epochs:
            print(f"Stopped: E stable at {rounded.tolist()} for {stable_epochs} epochs")
            break

        if epoch % 100 == 0: #It shows training progress every 100 epochs
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d} | Loss {total:.3e} | E {rounded.tolist()} | LR {lr:.1e}")

    elapsed = time.time() - t0
    print(f"Optimization completed in {elapsed:.1f} s")
    print(f"Epoch {epoch:4d} | Loss {total:.2e} | E {rounded.tolist()}")

    # final predicted curve
    with torch.no_grad():
        final_pred = model(inp)
        final_phys = problem.forward_physics(inp, final_pred)

    return model, history, inp, obs, final_pred, final_phys

# 6) Main: run indentation or vessel problem----------
if __name__ == '__main__':
    # Select problem: IndentationProblem (default) or VesselProblem
    # problem = IndentationProblem()
    problem = VesselProblem()  # ← Uncomment to switch to vessel inverse design

    # Design variable bounds (must match output_dim order from get_input_output_dims)
    # For Indentation: [E1_min, E1_max], [E2_min, E2_max], [E3_min, E3_max]
    if isinstance(problem, IndentationProblem):
        bounds = [(30, 250), (300, 1200), (30, 300)]
    else:
        # For Vessel: [SAngle, Stepply, Nrplies, SymLam, Thickpl]
        bounds = [(0, 175), (5, 55), (8, 40), (0, 1), (1, 2)]

    # Adjust data path based on problem type
    data_path = 'data/data.csv' if isinstance(problem, IndentationProblem) else 'data/pressure_vessel_DS.csv'
    
    # ========= PGNN Architecture Configuration =========
    # Customize the number of hidden layers and their size
    num_hidden_layers = 3      # ← Change this (e.g., 2, 3, 4, 5, ...)
    hidden_dim = 72            # ← Change this (e.g., 32, 64, 72, 128, ...)

    model, history, inp, obs, predictions, physics_output = train(
        problem=problem,
        bounds=bounds,
        data_path=data_path,
        num_hidden_layers=num_hidden_layers,  # Pass to train
        hidden_dim=hidden_dim,                # Pass to train
        patience=250,
        tighten_epochs=1500,
        stable_epochs=6
    )

    # Output & plots---------------------------
    Evals_int = predictions.mean(dim=0).cpu().numpy().round().astype(int)
    
    # Create output directory with problem-specific naming
    if isinstance(problem, IndentationProblem):
        output_dir = os.path.join(
            OUTPUT_ROOT,
            f"results_{Evals_int[0]}_{Evals_int[1]}_{Evals_int[2]}"
        )
    else:
        output_dir = os.path.join(
            OUTPUT_ROOT,
            f"vessel_opt_{Evals_int[0]}_{Evals_int[1]}_{Evals_int[2]}"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix permissions on output directory (for Docker/shared environments)
    try:
        os.chmod(output_dir, 0o777)
    except:
        pass
    
    print(f"Results will be saved to: {output_dir}/")

    # Save plots for Indentation problem only
    if isinstance(problem, IndentationProblem):
        δobs = problem._δobs      # (N,1)
        Fobs = problem._Fobs      # (N,1)
        Fpred_full = physics_output  # (N,1)

        # Save CSV and plots using visualization module
        save_predictions_csv(Fpred_full, δobs, output_dir, Evals_int)
        plot_force_indentation(δobs, Fobs, Fpred_full, output_dir)
        plot_loss_curves(history, output_dir)
    else:
        # For Vessel problem, just save loss curves
        plot_loss_curves(history, output_dir)

    print("Done.")
