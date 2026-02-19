# vessel.py
from physics.base import PhysicsProblem
import torch
import pandas as pd

class VesselProblem(PhysicsProblem):
    """
    Inverse-design problem for pressure vessel optimization.
    Objective: Find design parameters that minimize the objective value.
    Method: Hard nearest-neighbor lookup in dataset.
    """

    def __init__(self):
        self._inp = None
        self._obs = None

        self.design_params = [
            'SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl'
        ]

        self.bounds = [
            (0, 175), (8, 40), (5, 55), (0, 1), (1.0, 2.0)
        ]

        
        self.X = None  # [N, 5] design tensor
        self.y_min_val = None  # [N] min_val objectives
        self.y_S11 = None  # [N] S11 values
        self.y_S22 = None  # [N] S22 values
        self.y_Thick = None  # [N] Thick values
        self.min_val_min = None
        self.S11_min = None
        self.S22_min = None
        self.Thick_min = None
        self.steps = None  # Step size for each parameter (for rounding)

        self.best_design_found = None
        self.best_objective_found = float('inf')

    def get_input_output_dims(self):
        # Dummy input (not used) + 5 design parameters output
        return 1, 5

    def is_discrete(self):
        return True

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        X = torch.tensor(df[self.design_params].values, dtype=torch.float32)
        min_val = torch.tensor(df["min_val"].values, dtype=torch.float32)
        S11 = torch.tensor(df["S11"].values, dtype=torch.float32)
        S22 = torch.tensor(df["S22"].values, dtype=torch.float32)
        Thick = torch.tensor(df["Thick"].values, dtype=torch.float32)

        # Normalize each objective to start from 0 (best case)
        self.min_val_min = min_val.min().item()
        self.S11_min = S11.min().item()
        self.S22_min = S22.min().item()
        self.Thick_min = Thick.min().item()
        
        self.y_min_val = min_val - self.min_val_min
        self.y_S11 = S11 - self.S11_min
        self.y_S22 = S22 - self.S22_min
        self.y_Thick = Thick - self.Thick_min

        self.X = X
        self.steps = []
        for i, param in enumerate(self.design_params):
            unique_vals = sorted(df[param].unique())
            if len(unique_vals) > 1:
                diffs = [unique_vals[j+1] - unique_vals[j] for j in range(len(unique_vals)-1)]
                step = min(diffs)
            else:
                step = 1.0
            self.steps.append(step)

        best_idx = torch.argmin(min_val).item()
        self.best_design_found = dict(zip(self.design_params, X[best_idx].tolist()))
        self.best_objective_found = self.min_val_min

        print(f"\n[VesselProblem] Loaded dataset:")
        print(f"  Size: {len(df)}")
        print(f"  Best min_val (original): {self.min_val_min:.6f}")
        print(f"  Best S11 (original): {self.S11_min:.6f}")
        print(f"  Best S22 (original): {self.S22_min:.6f}")
        print(f"  Best Thick (original): {self.Thick_min:.6f}")
        print(f"  Best design: {self.best_design_found}")
        print(f"  Parameter steps: {dict(zip(self.design_params, self.steps))}")

        inp = torch.zeros(1, 1)  # Dummy input
        obs = torch.zeros(1, 4)  # Target: all objectives = 0 [min_val, S11, S22, Thick]
        self._inp = inp
        self._obs = obs
        return inp, obs

    def forward_physics(self, inp, predictions):
        """
        Soft nearest-neighbor lookup for [min_val, S11, S22, Thick].
        Returns predictions for all four objectives to enable multi-objective optimization.
        This is differentiable - gradients flow back to predictions.
        """
        # Normalize both for fair distance computation
        X_norm = self.X.clone()
        pred_norm = predictions.clone()
        for i in range(len(self.bounds)):
            lo, hi = self.bounds[i]
            X_norm[:, i] = (self.X[:, i] - lo) / (hi - lo)
            pred_norm[:, i] = (predictions[:, i] - lo) / (hi - lo)

        # Compute distances to all dataset points
        diff = pred_norm.unsqueeze(1) - X_norm.unsqueeze(0)  # [B, N, 5]
        dist = torch.norm(diff, dim=2)  # [B, N]

        # Soft nearest-neighbor: weighted average
        alpha = 10.0  # Higher alpha -> closer to hard nearest neighbor
        weights = torch.softmax(-alpha * dist, dim=1)  # [B,N]
        
        # Predict all four objectives
        y_min_val_pred = torch.sum(weights * self.y_min_val.unsqueeze(0), dim=1)
        y_S11_pred = torch.sum(weights * self.y_S11.unsqueeze(0), dim=1)
        y_S22_pred = torch.sum(weights * self.y_S22.unsqueeze(0), dim=1)
        y_Thick_pred = torch.sum(weights * self.y_Thick.unsqueeze(0), dim=1)

        # Track best solution found (with rounding for reporting)
        nearest_idx = torch.argmin(dist, dim=1)
        pred_rounded = self.X[nearest_idx]
        
        min_val = y_min_val_pred.min().item()
        if min_val < self.best_objective_found:
            idx = torch.argmin(y_min_val_pred).item()
            self.best_objective_found = min_val
            self.best_design_found = dict(
                zip(self.design_params, pred_rounded[idx].detach().tolist())
            )

        # Stack all four objectives for multi-objective optimization
        predictions_stacked = torch.stack([
            y_min_val_pred, y_S11_pred, y_S22_pred, y_Thick_pred
        ], dim=1)  # [B, 4]

        return predictions_stacked

    def constraint_loss(self, predictions):
        """Penalty if designs go outside valid bounds."""
        penalty = 0.0
        for i, (lo, hi) in enumerate(self.bounds):
            penalty += torch.mean(
                torch.relu(lo - predictions[:, i])**2 +
                torch.relu(predictions[:, i] - hi)**2
            )
        return penalty

    def get_best_design(self):
        return self.best_design_found
