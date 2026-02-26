from physics.base import PhysicsProblem
import torch
import pandas as pd
import numpy as np
import math
from scipy.optimize import curve_fit


class IndentationProblem(PhysicsProblem):
    """
    Forward problem: predict material elastic moduli from indentation data.

    CSV data: indentation depth (mm) → measured force (mF).
    NN predicts: E1, E2, E3 (elastic moduli in MPa).
    Physics: stiffness_calc() computes force from predicted moduli + depth.
    Loss: MSE between computed force and observed force.
    """

    def __init__(self):
        # ── Fixed physical constants (not predicted by the NN) ───────
        self.h1, self.h2, self.h3 = 0.6e-3, 1.7e-3, 52.6e-3   # layer thicknesses (m)
        self.EInd = 210e6       # indenter elastic modulus (Pa)
        self.R = 5e-3           # indenter radius (m)
        self.Rworm = 28e-3      # worm radius (m)
        self.nu = 0.45          # material Poisson's ratio
        self.nuInd = 0.33       # indenter Poisson's ratio
        self.stif_fit = None    # fitted power-law coefficient (set in load_data)

        # Cached data for plotting in save_results 
        self._targets = None       # observed force values
        self._input_data = None    # indentation depth values

        # Design parameters the NN will predict 
        self.design_params = ['E1', 'E2', 'E3']

        # Bounds: one (min, max) per design parameter 
        self.bounds = [
            (30, 250),      # E1 (MPa)
            (300, 1200),    # E2 (MPa)
            (30, 300),      # E3 (MPa)
        ]

    def get_input_output_dims(self):
        # 2 input columns: [observed_force, indentation_depth]
        # output_dim auto-sized from design_params
        return 2, len(self.design_params)

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        return 'data/data.csv'

    def load_data(self, path):
        df = pd.read_csv(path)
        df['ind'] = pd.to_numeric(df['ind'], errors='coerce')
        df['force'] = pd.to_numeric(df['force'], errors='coerce')

        ind = df['ind'].values * 1e-3      # mm → m
        force = df['force'].values * 1e-3   # mF → F

        # Fit power law F = a * δ^exp to get stiffness coefficient
        popt, _ = curve_fit(self.func, ind, force)
        self.stif_fit = popt

        targets = torch.tensor(force, dtype=torch.float32).unsqueeze(1)      # observed force [N, 1]
        input_data = torch.tensor(ind, dtype=torch.float32).unsqueeze(1)      # indentation depth [N, 1]

        # Cache for save_results
        self._targets = targets
        self._input_data = input_data

        # NN input: [observed_force, indentation_depth] — both columns
        inputs = torch.cat([targets, input_data], dim=1)
        return inputs, targets

    def func(self, x, a):
        maxX = max(x)
        eXp = 1.5 if maxX > 0.5 else 2.0
        return a * x**eXp

    def forward_physics(self, inputs, predictions):
        # Extract indentation depth from column 1 of inputs
        indentation = inputs[:, 1:2]
        computed, _ = self.stiffness_calc(predictions, indentation)
        return computed

    def constraint_loss(self, predictions):
        # E2 (predictions[:, 1]) must be >= E3 (predictions[:, 2])
        return torch.mean(torch.relu(predictions[:, 2] - predictions[:, 1])**2)

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        from visualization.plotting import save_predictions_csv, plot_force_indentation, plot_loss_curves

        input_data = self._input_data        # (N,1) indentation depth
        observed = self._targets              # (N,1) observed force

        param_means_int = predictions.mean(dim=0).cpu().numpy().round().astype(int)

        save_predictions_csv(computed_output, input_data, output_dir, param_means_int)
        plot_force_indentation(input_data, observed, computed_output, output_dir)
        plot_loss_curves(history, output_dir)

    def stiffness_calc(self, predictions, indentation):
        """
        Core physics: computes force from predicted material parameters + indentation depth.

        Args:
            predictions: [batch, output_dim] — NN predicted design parameters
            indentation: [batch, 1] — indentation depth values
        Returns:
            (force, k_eff) — computed force [batch, 1] and effective stiffness
        """
        # Unpack predicted parameters by column index
        E1, E2, E3 = predictions[:, 0], predictions[:, 1], predictions[:, 2]

        # Fixed physical constants from self
        h1, h2, h3 = self.h1, self.h2, self.h3
        EInd = self.EInd
        R, Rworm = self.R, self.Rworm

        L, pi = 1, math.pi
        R1, r1 = Rworm, Rworm - h1
        R2, r2 = r1, r1 - h2
        R3 = h3 / 2

        I1 = pi * (R1**4 - r1**4) / 4
        I2 = pi * (R2**4 - r2**4) / 4
        I3 = pi * (R3**4) / 4

        maxX = indentation.max().item()
        if maxX > 0.43:
            eXp, c1 = 1.5, 5.9
        else:
            eXp, c1 = 2.0, 9.3

        kind = 2 * pi * EInd * R

        k1 = E1 * I1 / L**4
        k2 = E2 * I2 / L**4
        k3 = E3 * I3 / L**4

        ta = np.array([h1 * 1000, h2 * 1000, h3 * 1000])
        uy = maxX / 2
        khy = (uy + ta) * (2 * uy**2 + 2 * ta * uy + ta**2) / (2 * uy + ta)**3
        hy1, hy2, hy3 = khy if maxX >= 0.7 else [1, 1, 1]

        kstruct = c1 * (hy1 * k1 + hy2 * k2 + hy3 * k3) * 1000
        k_eff = 1 / (1 / kind + 1 / kstruct)

        return k_eff * indentation**eXp, k_eff
