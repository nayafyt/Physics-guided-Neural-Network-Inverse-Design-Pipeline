from physics.base import PhysicsProblem
import torch
import pandas as pd
import numpy as np
import math
from scipy.optimize import curve_fit


class IndentationProblem(PhysicsProblem):

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
        return 2, 3

    def get_bounds(self):
        """Returns bounds for [E1, E2, E3] parameters (MPa)"""
        return [(30, 250), (300, 1200), (30, 300)]

    def get_data_path(self):
        """Returns path to indentation data"""
        return 'data/data.csv'

    
    def load_data(self, path):
        df = pd.read_csv(path)
        df['ind'] = pd.to_numeric(df['ind'], errors='coerce')
        df['force'] = pd.to_numeric(df['force'], errors='coerce')

        ind = df['ind'].values * 1e-3
        force = df['force'].values * 1e-3

        popt, _ = curve_fit(self.func, ind, force)
        self.stif_fit = popt

        Fobs = torch.tensor(force, dtype=torch.float32).unsqueeze(1)
        δobs = torch.tensor(ind, dtype=torch.float32).unsqueeze(1)

        self._Fobs = Fobs
        self._δobs = δobs

        inp = torch.cat([Fobs, δobs], dim=1)
        return inp, Fobs

    def func(self, x, a):
        maxX = max(x)
        eXp = 1.5 if maxX > 0.5 else 2.0
        return a * x**eXp

 
    def forward_physics(self, inp, predictions):
        δ = inp[:, 1:2]
        E1 = predictions[:, 0]
        E2 = predictions[:, 1]
        E3 = predictions[:, 2]

        Fpred, _ = self.stiffness_calc(
            self.EInd, self.nu, self.nuInd, self.R,
            E1, E2, E3,
            self.Rworm, self.h1, self.h2, self.h3,
            δ, self.stif_fit
        )
        return Fpred


    def constraint_loss(self, predictions):
        E2 = predictions[:, 1]
        E3 = predictions[:, 2]
        return torch.mean(torch.relu(E3 - E2)**2)

    def save_results(self, history, epoch_results, output_dir, predictions, physics_output, inp, obs):
        """Custom result saving for indentation problem"""
        from visualization.plotting import save_predictions_csv, plot_force_indentation, plot_loss_curves
        
        δobs = self._δobs      # (N,1) - use problem's internal data
        Fobs = self._Fobs      # Observed force
        Fpred_full = physics_output  # (N,1)
        
        Evals_int = predictions.mean(dim=0).cpu().numpy().round().astype(int)
        
        # Save CSV and plots
        save_predictions_csv(Fpred_full, δobs, output_dir, Evals_int)
        plot_force_indentation(δobs, Fobs, Fpred_full, output_dir)
        plot_loss_curves(history, output_dir)

    def stiffness_calc(self, EInd, nu, nuInd, R,
                       E1, E2, E3, Rworm,
                       h1, h2, h3, δ, stif_fit):

        Rstar = 1/(1/R + 1/Rworm)

        L, pi = 1, math.pi
        R1, r1 = Rworm, Rworm - h1
        R2, r2 = r1, r1 - h2
        R3 = h3 / 2

        I1 = pi*(R1**4 - r1**4)/4
        I2 = pi*(R2**4 - r2**4)/4
        I3 = pi*(R3**4)/4

        maxX = δ.max().item()
        if maxX > 0.43:
            eXp, c1 = 1.5, 5.9
        else:
            eXp, c1 = 2.0, 9.3

        kind = 2 * pi * EInd * R

        k1 = E1 * I1 / L**4
        k2 = E2 * I2 / L**4
        k3 = E3 * I3 / L**4

        ta = np.array([h1*1000, h2*1000, h3*1000])
        uy = maxX / 2
        khy = (uy+ta)*(2*uy**2 + 2*ta*uy + ta**2)/(2*uy+ta)**3
        hy1, hy2, hy3 = khy if maxX >= 0.7 else [1,1,1]

        kstruct = c1 * (hy1*k1 + hy2*k2 + hy3*k3) * 1000
        k_eff = 1/(1/kind + 1/kstruct)

        return k_eff * δ**eXp, k_eff
