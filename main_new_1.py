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

Main inputs to train():
  path          : str  - CSV file path with columns 'ind' and 'force'
  patience      : int  - epochs of no improvement before early stopping
  tighten_epochs: int  - T_max for CosineAnnealingLR
  E_min_max     : list - bounds for E1, E2, E3 as [(min1,max1),...]
  stable_epochs : int  - number of consecutive epochs with no first-decimal change
"""
import time
import random
import numpy as np
import torch  #Open-source machine learning library
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
from scipy.optimize import curve_fit
import os

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

class PGNN(nn.Module):
    """Physics-Guided NN: input (force,indentation) -> predicted [E1,E2,E3]."""
    def __init__(self, E_min_max):
        super().__init__()
        # hidden layers: 2 -> 72 -> 72 -> 72 -> raw 3 outputs
        self.hidden = nn.Sequential(
            
            nn.Linear(2, 72), nn.BatchNorm1d(72), nn.Tanh(),
            nn.Linear(72,72), nn.BatchNorm1d(72), nn.Tanh(),
            nn.Linear(72,72), nn.BatchNorm1d(72), nn.Tanh(),
            nn.Linear(72,3)

            #nn.Linear(2, 92), nn.BatchNorm1d(92), nn.Tanh(),
            #nn.Linear(92,92), nn.BatchNorm1d(92), nn.Tanh(),
            #nn.Linear(92,3)
        )
        # per-output bounding activations
        self.act1 = CustomActivation(*E_min_max[0])
        self.act2 = CustomActivation(*E_min_max[1])
        self.act3 = CustomActivation(*E_min_max[2])
    def forward(self, x):
        raw = self.hidden(x)
        # apply bounds
        E1 = self.act1(raw[:,0].unsqueeze(1)).squeeze(1)
        E2 = self.act2(raw[:,1].unsqueeze(1)).squeeze(1)
        E3 = self.act3(raw[:,2].unsqueeze(1)).squeeze(1)
        return torch.stack([E1,E2,E3], dim=1)

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

def load_data(path):
    """Load CSV with 'ind','force'; convert to torch tensors in SI units."""
    df = pd.read_csv(path)
    df['ind']   = pd.to_numeric(df['ind'],   errors='coerce')
    df['force'] = pd.to_numeric(df['force'], errors='coerce')
    ind   = df['ind'].values * 1e-3   # mm -> m
    force = df['force'].values * 1e-3 # mN -> N
    popt,_ = curve_fit(func, ind, force)
    return (
        torch.tensor(force, dtype=torch.float32).unsqueeze(1),
        torch.tensor(ind,   dtype=torch.float32).unsqueeze(1),
        popt
    )

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

def generate_data(path):
    """Load observed data + generate synthetic 'true' curve (unused)."""
    torch.manual_seed(SEED)
    Fobs, δobs, stif = load_data(path)
    # dummy true moduli & geometry
    E1, E2, E3, EInd = 144, 2461, 1177, 210e6
    h1, h2, h3 = 0.6e-3, 1.7e-3, 52.6e-3
    nu, nuInd, R, Rworm = 0.45, 0.33, 5e-3, 28e-3
    _, _ = stiffness_calc(
        EInd, nu, nuInd, R,
        E1, E2, E3, Rworm,
        h1, h2, h3, δobs, stif
    )
    return Fobs, δobs, nu, nuInd, EInd, R, Rworm, h1, h2, h3, stif

# 4) LOSS & TRAINING ----------------------------------------------
def loss_fn(model, inp, Fobs, δobs,
            nu, nuInd, EInd, R, Rworm,
            h1, h2, h3, stif,
            wOrder):
    """Compute total loss = MSE(data) + wOrder * hinge(E3>E2)."""
    E_pred = model(inp)
    E1, E2, E3 = E_pred[:,0], E_pred[:,1], E_pred[:,2]
    Fpred, _ = stiffness_calc(
        EInd, nu, nuInd, R,
        E1, E2, E3, Rworm,
        h1, h2, h3, δobs, stif
    )
    data_loss  = torch.mean((Fobs - Fpred)**2)
    order_vio  = torch.mean(torch.relu(E3 - E2)**2)
    total_loss = data_loss + wOrder * order_vio
    return total_loss, data_loss, order_vio, E_pred, Fpred

def train(path, patience, tighten_epochs, E_min_max, stable_epochs=3):
    """
    Train PGNN on (force,delta) data in `path`, stopping when:
      - no loss improvement for `patience` epochs, or
      - predicted E means stable (first decimal) for `stable_epochs` epochs.

    Returns trained model.
    """
    t0 = time.time()

    # instantiate model
    model = PGNN(E_min_max)
    model.apply(init_weights)
    optimizer = optim.Adam(model.hidden.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=tighten_epochs, eta_min=1e-5)

    # load and prepare data
    Fobs, δobs, nu, nuInd, EInd, R, Rworm, h1, h2, h3, stif = generate_data(path)
    inp = torch.cat([Fobs, δobs], dim=1)  # [N,2]

    history = {'total':[], 'data':[], 'order':[]}
    stable_count = 0
    prev_rounded = None
    best_loss, no_imp = float('inf'), 0

    for epoch in range(1, 10001): # max 10k epochs
        optimizer.zero_grad()
        total, dL, oL, E_pred, Fpred = loss_fn(
            model, inp, Fobs, δobs,
            nu, nuInd, EInd, R, Rworm,
            h1, h2, h3, stif,
            wOrder=1.0
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # record
        history['total'].append(total.item())
        history['data'].append(dL.item())
        history['order'].append(oL.item())

        # check improvement
        if total.item() < best_loss:
            best_loss, no_imp = total.item(), 0
        else:
            no_imp += 1

        # first-decimal stability early-stop
        #On our run this got trigger first before no improvement condition
        E_means = E_pred.mean(dim=0)
        rounded = torch.round(E_means * 10) / 10  # first decimal
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
            print(f"Epoch {epoch:4d} | Loss {total:.2e} | E {rounded.tolist()} | LR {lr:.1e}")

    elapsed = time.time() - t0
    print(f"Optimization completed in {elapsed:.1f} s")
    print(f"Epoch {epoch:4d} | Loss {total:.2e} | E {rounded.tolist()}")

    # final predicted curve
    with torch.no_grad():
        E_final    = model(inp)
        Fpred_full, _ = stiffness_calc(
            EInd, nu, nuInd, R,
            E_final[:,0], E_final[:,1], E_final[:,2],
            Rworm, h1, h2, h3, δobs, stif
        )
    
    # Create output folder with predicted moduli
    Evals_int  = E_final.mean(dim=0).cpu().numpy().round().astype(int)
    output_dir = f"results_{Evals_int[0]}_{Evals_int[1]}_{Evals_int[2]}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}/")
    
    # Save CSV with predictions
    deltaex_np = δobs.squeeze(1).cpu().numpy()
    Fpred_np   = Fpred_full.squeeze(1).cpu().numpy()
    jobName = os.path.join(output_dir, f"Pred_{Evals_int[0]}_{Evals_int[1]}_{Evals_int[2]}.csv")

    with open(jobName, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['deltaex', 'Fpred'])
        for d,fp in zip(deltaex_np, Fpred_np):
            writer.writerow([d, fp])
    print(f"Saved predictions to {jobName}")

    # Save plots to output folder
    plt.figure()
    plt.plot(δobs, Fobs, 'bo', label='meas')
    plt.plot(δobs, Fpred_full, 'r-', label='pred')
    plt.xlabel("Indentation δ (m)")
    plt.ylabel("Force F (N)")
    plt.title("Measured vs Predicted Force–Indentation Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "force_indentation.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history['total'], label='total')
    plt.plot(history['data'], label='data')
    plt.plot(history['order'], label='order')
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=300)
    plt.close()

    return model

if __name__ == '__main__':
    
    model = train(
        path='data.csv',
        patience=250,
        tighten_epochs=1500,
        E_min_max=[(30,250),(300,1200),(30,300)],
        stable_epochs=6
    )
