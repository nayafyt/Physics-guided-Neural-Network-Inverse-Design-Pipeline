#!/usr/bin/env python3
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

# Import visualization utilities
from visualization import save_predictions_csv, plot_force_indentation, plot_loss_curves

# 1) DETERMINISM -------------------------------------------------- 
# ? Ensures reproducible results across runs.. 
# Meaning if we changed the seed every few runs, we would get different results, that will give a better idea of the performance of the model?
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  #if using multi-GPU
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

class GenericPGNN(nn.Module):
    """
    Generic Physics-Guided NN with configurable input/output dims and bounds.
    For your indentation problem: input_dim=2, output_dim=3.
    """
    def __init__(self, input_dim, output_dim, bounds):
        super().__init__()

        hidden_dim = 72

        layers = []
        prev = input_dim
        for _ in range(3):  # 3 hidden layers of size 72
            layers.append(nn.Linear(prev, hidden_dim))
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

# 3) DATA & PHYSICS HELPERS ---------------------------------------
def func(x, a):
    """Power-law fit function: exponent depends on max(x)."""
    maxX = max(x)
    eXp = 1.5 if maxX>0.5 else 2.0
    return a * x**eXp

def stiffness_calc(EInd, nu, nuInd, R,
                   E1, E2, E3, Rworm,
                   h1, h2, h3, δ, stif_fit):
    """
    Compute predicted force via series/parallel spring analogy
    and nonlinearity based on indentation regime.
    """
    # effective radius
    Rstar = 1/(1/R + 1/Rworm)
    # geometry
    L, pi = 1, math.pi
    R1, r1 = Rworm, Rworm - h1
    R2, r2 = r1, r1 - h2
    R3      = h3/2
    # moments of inertia
    I1 = pi*(R1**4 - r1**4)/4
    I2 = pi*(R2**4 - r2**4)/4
    I3 = pi*(R3**4)/4
    # exponent & constant by depth
    maxX = δ.max().item()
    if maxX > 0.43:
        eXp, c1 = 1.5, 5.9
    else:
        eXp, c1 = 2.0, 9.3
    # indenter stiffness
    kind = 2*pi*EInd*R
    # layer stiffnesses
    k1 = E1 * I1 / L**4
    k2 = E2 * I2 / L**4
    k3 = E3 * I3 / L**4
    # hyperelastic correction
    ta = np.array([h1*1000, h2*1000, h3*1000])
    uy = maxX/2
    khy = (uy+ta)*(2*uy**2 + 2*ta*uy + ta**2)/(2*uy+ta)**3
    hy1, hy2, hy3 = khy if maxX>=0.7 else [1,1,1]
    kstruct = c1*(hy1*k1 + hy2*k2 + hy3*k3)*1000
    # series
    k_eff = 1/(1/kind + 1/kstruct)
    # predicted force
    return k_eff * δ**eXp, k_eff

# 4) Physics problem------------------------------

class IndentationPhysicsProblem:
    def __init__(self):
        self.h1, self.h2, self.h3 = 0.6e-3, 1.7e-3, 52.6e-3
        self.EInd = 210e6
        self.R = 5e-3
        self.Rworm = 28e-3
        self.nu = 0.45
        self.nuInd = 0.33
        self.stif_fit = None
        self._Fobs = None
        self._δobs = None

    def get_input_output_dims(self):
        """This problem has 2 inputs (force, indentation) and 3 outputs (E1,E2,E3)."""
        return 2, 3

    def load_data(self, path):
        """Load CSV with 'ind','force'; convert to torch tensors in SI units."""
        df = pd.read_csv(path)
        df['ind'] = pd.to_numeric(df['ind'], errors='coerce')
        df['force'] = pd.to_numeric(df['force'], errors='coerce')
        ind = df['ind'].values * 1e-3  # mm -> m
        force = df['force'].values * 1e-3  # mN -> N

        popt, _ = curve_fit(func, ind, force)
        self.stif_fit = popt

        # Construct input exactly as original: torch.cat([Fobs, δobs], dim=1)
        Fobs = torch.tensor(force, dtype=torch.float32).unsqueeze(1)
        δobs = torch.tensor(ind, dtype=torch.float32).unsqueeze(1)
        inp = torch.cat([Fobs, δobs], dim=1)  # [N, 2]
        obs = Fobs  # observed force

        # store for plotting
        self._Fobs = Fobs
        self._δobs = δobs

        return inp, obs

    def forward_physics(self, inp, predictions):
        """Compute predicted force from E1, E2, E3."""
        # indentation is second column
        δ = inp[:, 1:2]  # shape (N,1)
        δ_flat = δ.squeeze(1)  # shape (N,)
        E1 = predictions[:, 0]
        E2 = predictions[:, 1]
        E3 = predictions[:, 2]

        Fpred, _ = stiffness_calc(
            self.EInd, self.nu, self.nuInd, self.R,
            E1, E2, E3, self.Rworm,
            self.h1, self.h2, self.h3,
            δ_flat, self.stif_fit
        )
        return Fpred.unsqueeze(1)

    def constraint_loss(self, predictions):
        """Hinge penalty enforcing E2 >= E3."""
        E2 = predictions[:, 1]
        E3 = predictions[:, 2]
        return torch.mean(torch.relu(E3 - E2)**2)

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
          patience=250, tighten_epochs=1500, stable_epochs=6):
    """
    Generic training loop for any PhysicsProblem.
    """
    t0 = time.time()

    input_dim, output_dim = problem.get_input_output_dims()
    model = GenericPGNN(input_dim, output_dim, bounds)
    model.apply(init_weights)
    optimizer = optim.Adam(model.hidden.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=tighten_epochs, eta_min=1e-5)

    inp, obs = problem.load_data(data_path)
    print(f"Input shape: {inp.shape}")
    print(f"Input first row: {inp[0]}")
    print(f"Obs shape: {obs.shape}")
    print(f"Stif_fit: {problem.stif_fit}")
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

# 6) Main: run indentation as a PhysicsProblem----------------

if __name__ == '__main__':
    problem = IndentationPhysicsProblem()

    # generic bounds (can be changed per physics problem)
    bounds = [(30, 250), (300, 1200), (30, 300)]

    model, history, inp, obs, predictions, physics_output = train(
        problem=problem,
        bounds=bounds,
        data_path='data.csv',
        patience=250,
        tighten_epochs=1500,
        stable_epochs=6
    )

    # Output & plots---------------------------
    Evals_int = predictions.mean(dim=0).cpu().numpy().round().astype(int)
    output_dir = f"results_{Evals_int[0]}_{Evals_int[1]}_{Evals_int[2]}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")

    δobs = problem._δobs      # (N,1)
    Fobs = problem._Fobs      # (N,1)
    Fpred_full = physics_output  # (N,1)

    # Save CSV and plots using visualization module
    save_predictions_csv(Fpred_full, δobs, output_dir, Evals_int)
    plot_force_indentation(δobs, Fobs, Fpred_full, output_dir)
    plot_loss_curves(history, output_dir)

    print("Done.")
