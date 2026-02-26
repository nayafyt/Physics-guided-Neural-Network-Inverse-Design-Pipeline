"""
Physics-Guided Neural Network (PGNN) inverse design pipeline with:
  - Deterministic setup (fixed random seeds)
  - Cosine-annealed learning rate for exploration/exploitation
  - Constraint loss for enforcing parameter ordering/relationships
  - Early stopping once mean predicted parameters stabilize at first decimal
    (configurable number of consecutive stable epochs)
  - CSV export of final predictions
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
from visualization.plotting import save_predictions_csv, save_vessel_design_csv, save_vessel_epoch_results_csv, plot_force_indentation, plot_loss_curves, evaluate_rank

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
    """Maps raw network outputs to [out_min, out_max] via tanh scaling."""
    def __init__(self, out_min, out_max):
        super().__init__()
        self.out_min, self.out_max = out_min, out_max
    def forward(self, x):
        # tanh in [-1,1] -> scale to [out_min, out_max]
        return self.out_min + (self.out_max - self.out_min) * (torch.tanh(x) + 1) / 2

class PGNN(nn.Module):
    """
    Generic Physics-Guided NN with configurable input/output dims and bounds.
    - For indentation (forward): input_dim=2, output_dim=3
    - For inverse design: input_dim=1, output_dim=5 (maps target objective → design params)
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension (number of parameters to predict)
        bounds (list): List of (min, max) tuples for each output
        num_hidden_layers (int): Number of hidden layers (default: 3)
        hidden_dim (int): Size of each hidden layer (default: 72)
        use_layer_norm (bool): If True use LayerNorm (needed when batch_size=1),
                               if False use BatchNorm (needs batch_size>1)
    """
    def __init__(self, input_dim, output_dim, bounds, num_hidden_layers=3, hidden_dim=72, use_layer_norm=False):
        super().__init__()

        layers = []
        prev = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            else:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.hidden = nn.Sequential(*layers)

        self.activations = nn.ModuleList([
            CustomActivation(bounds[i][0], bounds[i][1])
            for i in range(output_dim)
        ])

    def forward(self, x):
        raw = self.hidden(x)
        outputs = []
        for i in range(len(self.activations)):
            activated = self.activations[i](raw[:, i].unsqueeze(1)).squeeze(1)  # one predicted parameter, squeezed into its allowed range ([out_min, out_max]) using tanh
            outputs.append(activated)
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

def loss_fn(model, inputs, targets, problem, constraint_weight=1.0):
    """
    Generic loss:
        total = MSE(targets, computed_output) + constraint_weight * constraint
    """
    predictions = model(inputs)
    computed_output = problem.forward_physics(inputs, predictions)  # physics simulation result from predicted params
    data_loss = torch.mean((targets - computed_output)**2)
    constraint = problem.constraint_loss(predictions)  # penalty for violating parameter relationships
    total_loss = data_loss + constraint_weight * constraint
    return total_loss, data_loss, constraint, predictions, computed_output

def train(problem, bounds, data_path,
          num_hidden_layers, hidden_dim,
          patience, tighten_epochs, stable_epochs):
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

    # Load data first to determine batch_size (needed to pick LayerNorm vs BatchNorm)
    inputs, targets = problem.load_data(data_path)  # targets = ground-truth data the model tries to match
    batch_size = inputs.shape[0]

    input_dim, output_dim = problem.get_input_output_dims()
    # BatchNorm needs batch_size>1 to compute variance; use LayerNorm when batch_size=1
    use_layer_norm = (batch_size == 1)
    model = PGNN(input_dim, output_dim, bounds, num_hidden_layers=num_hidden_layers, hidden_dim=hidden_dim, use_layer_norm=use_layer_norm)
    model.apply(init_weights)

    optimizer = optim.Adam(model.hidden.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=tighten_epochs, eta_min=1e-5)
    # eta_min controls the minimum learning rate: higher = warmer end (more exploration),
    # lower = colder end (more refinement). Try 1e-4 or 1e-6 to adjust late-stage optimization.
    history = {'total': [], 'data': [], 'constraint': []}
    epoch_results = []  # Store predictions and computed outputs for each epoch
    best_loss = float('inf')
    no_improvement = 0
    stable_count = 0
    prev_rounded = None
    last_saved_predictions = None  # Track last saved prediction

    for epoch in range(1, tighten_epochs + 1):
        optimizer.zero_grad()
        # predictions = NN predicted design params, computed = physics simulation output from those params
        # constraint = penalty for violating parameter relationships
        total_loss, data_loss, constraint, predictions, computed = loss_fn(model, inputs, targets, problem, constraint_weight=1.0)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history['total'].append(total_loss.item())
        history['data'].append(data_loss.item())
        history['constraint'].append(constraint.item())
        
        # Store epoch results for vessel problem - save if predictions changed or every 10 epochs
        if isinstance(problem, VesselProblem):
            predictions_np = predictions.detach().cpu().numpy().copy()  # numpy copy of predicted params
            computed_np = computed.detach().cpu().numpy().copy()  # numpy copy of physics simulation output

            # Check if predictions have changed significantly or if it's a milestone epoch
            save_this_epoch = (epoch % 10 == 0)  # Save every 10 epochs
            if last_saved_predictions is not None:
                prediction_diff = np.abs(predictions_np - last_saved_predictions).max()
                if prediction_diff > 0.01:  # Save if change > 0.01
                    save_this_epoch = True

            if save_this_epoch or epoch == 1:  # Always save epoch 1
                epoch_results.append({
                    'epoch': epoch,
                    'predictions': predictions_np,
                    'physics_output': computed_np
                })
                last_saved_predictions = predictions_np

        # check improvement
        if total_loss.item() < best_loss:
            best_loss, no_improvement = total_loss.item(), 0
        else:
            no_improvement += 1

        # first-decimal stability early-stop
        param_means = predictions.mean(dim=0)  # average of each predicted parameter across the batch
        rounded = torch.round(param_means * 10) / 10  # first decimal
        if prev_rounded is not None and torch.all(rounded == prev_rounded):
            stable_count += 1
        else:
            stable_count = 0
        prev_rounded = rounded

        # stopping conditions
        if no_improvement >= patience:
            print(f"Early stop (no loss imp.) @ epoch {epoch}")
            break
        if stable_count >= stable_epochs:
            print(f"Stopped: params stable at {rounded.tolist()} for {stable_epochs} epochs")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss {total_loss:.3e} | Params {rounded.tolist()}")

    elapsed = time.time() - t0
    print(f"Optimization completed in {elapsed:.1f} s")
    print(f"Epoch {epoch:4d} | Loss {total_loss:.2e} | Params {rounded.tolist()}")

    # final predictions
    with torch.no_grad():
        final_predictions = model(inputs)  # final predicted design params after training
        final_computed = problem.forward_physics(inputs, final_predictions)  # physics simulation output from final params

    # Return results as dict for dynamic unpacking (supports any problem type)
    result = {
        'model': model,
        'history': history,
        'inputs': inputs,
        'targets': targets,
        'predictions': final_predictions,
        'computed_output': final_computed,
        'epoch_results': epoch_results if isinstance(problem, VesselProblem) else None
    }
    return result

# 6) Main: run indentation or vessel problem----------
if __name__ == '__main__':
    # SELECT YOUR PHYSICS PROBLEM
    # Uncomment ONE of the following:
    problem = IndentationProblem()          # Inverse indentation problem (default)
    # problem = VesselProblem()              # Pressure vessel optimization
    # problem = YourProblem()                # Your custom physics problem

    # Get bounds and data path dynamically from the problem
    bounds = problem.get_bounds()
    data_path = problem.get_data_path() #objective value for each physics problem
    
    # NEURAL NETWORK ARCHITECTURE - TUNE THESE FOR YOUR PROBLEM
    num_hidden_layers = 3      # ← Try: 2, 3, 4, 5 (more layers = more capacity)
    hidden_dim = 72            # ← Try: 32, 64, 72, 128 (wider = more expressive)

    # TRAINING HYPERPARAMETERS - ADJUST TO CONTROL OPTIMIZATION
    patience = 250             # ← Early stopping: stop if loss doesn't improve for N epochs
    tighten_epochs = 10000     # ← Maximum training epochs (upper bound on total epochs)
    stable_epochs = 6          # ← Stop if predictions stable for N consecutive epochs
    
    # How to tune based on results:
    # - If model hasn't converged: increase tighten_epochs or patience
    # - If stopping too early: increase patience or stable_epochs
    # - If model underfitting: add layers (num_hidden_layers) or increase hidden_dim
    # - If model overfitting or too slow: reduce layers/hidden_dim
    # - If optimization stuck in local minima: decrease patience to allow longer search
    
    result = train(
        problem=problem,
        bounds=bounds,
        data_path=data_path,
        num_hidden_layers=num_hidden_layers,  
        hidden_dim=hidden_dim,              
        patience=patience,
        tighten_epochs=tighten_epochs,
        stable_epochs=stable_epochs
    )
    
    # Dynamically unpack result dict (works for any problem type)
    model = result['model']
    history = result['history']
    inputs = result['inputs']
    targets = result['targets']
    predictions = result['predictions']
    computed_output = result['computed_output']
    epoch_results = result['epoch_results']

    # Output & plots---------------------------
    param_means_int = predictions.mean(dim=0).cpu().numpy().round().astype(int)  # mean of each param, rounded to int
    
    # Create output directory dynamically using problem's naming scheme
    output_dir_name = problem.get_output_dir_name(predictions)
    output_dir = os.path.join(OUTPUT_ROOT, output_dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix permissions on output directory (for Docker/shared environments)
    try:
        os.chmod(output_dir, 0o777)
    except:
        pass
    
    print(f"Results will be saved to: {output_dir}/")

    # Call problem-specific result saving (handles all visualization)
    problem.save_results(history, epoch_results, output_dir, predictions, computed_output, inputs, targets)

    print("Done.")
