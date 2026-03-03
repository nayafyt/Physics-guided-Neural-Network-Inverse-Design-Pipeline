from physics.base import PhysicsProblem
import torch
import numpy as np
import pandas as pd

# Available objective modes for multi-objective optimization
OBJECTIVE_MODES = ['min_val', 'tchebycheff', 'product', 'pnorm']


class VesselProblem(PhysicsProblem):
    """
    Inverse design: find pressure vessel parameters that minimize the objective.

    CSV data: design-space with parameter combos and their objective (min_val).
    NN predicts: SAngle, Nrplies, Stepply, SymLam, Thickpl.
    Physics: soft nearest-neighbor lookup in the dataset (differentiable).
    Loss: drives the NN toward the design with the lowest objective.

    Supports multiple objective modes:
        - 'min_val':      Original pre-computed objective (default)
        - 'tchebycheff':  Minimizes the worst of S11/2500, S22/185, Thick (normalized to [0,1])
        - 'product':      Minimizes S11/2500 * S22/185 * Thick (all must be small)
        - 'pnorm':        Smooth approximation to Tchebycheff using L4 norm
    """

    def __init__(self, objective_mode='min_val', alpha=10.0):
        """
        Args:
            objective_mode: How to compute the objective. One of:
                'min_val'     - use pre-computed min_val column (default)
                'tchebycheff' - minimize max(S11/2500, S22/185, Thick) normalized to [0,1]
                'product'     - minimize S11/2500 * S22/185 * Thick
                'pnorm'       - minimize (S11^4 + S22^4 + Thick^4)^(1/4) normalized to [0,1]
            alpha: Softmax sharpness for nearest-neighbor lookup.
                Higher = closer to hard nearest-neighbor (default: 10.0)
        """
        if objective_mode not in OBJECTIVE_MODES:
            raise ValueError(f"objective_mode must be one of {OBJECTIVE_MODES}, got '{objective_mode}'")

        # Cached data for save_results
        self._inputs = None
        self._targets = None

        self.objective_mode = objective_mode
        self.alpha = alpha

        # Design parameters the NN will predict
        self.design_params = [
            'SAngle',       # spiral angle (0–175°)
            'Nrplies',      # number of plies (8–40)
            'Stepply',      # step ply (5–55)
            'SymLam',       # symmetric laminate (0–1, discrete)
            'Thickpl',      # ply thickness (1.0–2.0 mm)
        ]

        # Bounds: one (min, max) per design parameter
        self.bounds = [
            (0, 175), (8, 40), (5, 55), (0, 1), (1.0, 2.0)
        ]

        # Dataset state (populated in load_data)
        self.X = None               # [N, output_dim] design parameter tensor
        self.y = None               # [N] objective values (shifted so best = 0)
        self.min_val_min = None     # original minimum objective value
        self.steps = None           # step size per parameter (for rounding)

        self.best_design_found = None
        self.best_objective_found = float('inf')

    def get_input_output_dims(self):
        # Dummy input (not used) + output_dim auto-sized from design_params
        return 1, len(self.design_params)

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        return 'data/pressure_vessel_DS.csv'

    def _compute_objective(self, df):
        """Compute objective values based on the selected mode."""
        s11 = df["S11"].values
        s22 = df["S22"].values
        thick = df["Thick"].values

        if self.objective_mode == 'min_val':
            return df["min_val"].values

        # Normalize S11/2500, S22/185 to bring them to comparable scale
        o1 = s11 / 2500.0
        o2 = s22 / 185.0
        o3 = thick

        if self.objective_mode == 'tchebycheff':
            # Normalize each to [0,1], then take max → minimizes the worst objective
            o1n = (o1 - o1.min()) / (o1.max() - o1.min())
            o2n = (o2 - o2.min()) / (o2.max() - o2.min())
            o3n = (o3 - o3.min()) / (o3.max() - o3.min())
            return np.maximum.reduce([o1n, o2n, o3n])

        elif self.objective_mode == 'product':
            # Product: all three must be small for the product to be small
            return o1 * o2 * o3

        elif self.objective_mode == 'pnorm':
            # L4 norm of normalized objectives: smooth approximation to max
            o1n = (o1 - o1.min()) / (o1.max() - o1.min())
            o2n = (o2 - o2.min()) / (o2.max() - o2.min())
            o3n = (o3 - o3.min()) / (o3.max() - o3.min())
            p = 4
            return (o1n**p + o2n**p + o3n**p) ** (1/p)

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        X = torch.tensor(df[self.design_params].values, dtype=torch.float32)

        obj_values = self._compute_objective(df)
        min_val = torch.tensor(obj_values, dtype=torch.float32)

        self.min_val_min = min_val.min().item()
        y = min_val - self.min_val_min  # shift so best design has objective = 0

        self.X = X
        self.y = y

        # Compute step size per parameter (minimum gap between unique values)
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
        print(f"  Objective mode: {self.objective_mode} | Alpha: {self.alpha}")
        print(f"  Size: {len(df)}")
        print(f"  Best objective: {self.min_val_min:.6f}")
        print(f"  Best design: {self.best_design_found}")
        print(f"  Parameter steps: {dict(zip(self.design_params, self.steps))}")

        # Dummy tensors — forward_physics uses self.X / self.y instead
        inputs = torch.zeros(1, 1)
        targets = torch.zeros(1, 1)
        self._inputs = inputs
        self._targets = targets
        return inputs, targets

    def forward_physics(self, inputs, predictions):
        """
        Soft nearest-neighbor lookup: differentiable so gradients flow back.
        Computes weighted average of objectives based on distance to dataset.
        """
        # Normalize both predicted and dataset params to [0, 1] for fair distance
        X_norm = self.X.clone()
        pred_norm = predictions.clone()
        for i, (lo, hi) in enumerate(self.bounds):
            X_norm[:, i] = (self.X[:, i] - lo) / (hi - lo)
            pred_norm[:, i] = (predictions[:, i] - lo) / (hi - lo)

        # Compute distances to all dataset points
        diff = pred_norm.unsqueeze(1) - X_norm.unsqueeze(0)  # [B, N, output_dim]
        dist = torch.norm(diff, dim=2)  # [B, N]

        # Soft nearest-neighbor via softmax weights
        alpha = self.alpha  # higher = closer to hard nearest-neighbor
        weights = torch.softmax(-alpha * dist, dim=1)          # [B, N]
        y_pred = torch.sum(weights * self.y.unsqueeze(0), dim=1)  # [B]

        # Track best solution found
        # Find which dataset point each prediction is closest to
        nearest_idx = torch.argmin(dist, dim=1)
        pred_rounded = self.X[nearest_idx]

        min_val = y_pred.min().item()
        if min_val < self.best_objective_found:
            idx = torch.argmin(y_pred).item()
            self.best_objective_found = min_val
            self.best_design_found = dict(
                zip(self.design_params, pred_rounded[idx].detach().tolist())
            )

        return y_pred.unsqueeze(1)

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

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        import os
        from visualization.plotting import plot_loss_curves, save_vessel_epoch_results_csv, evaluate_rank

        if epoch_results is not None:
            epoch_csv_path = os.path.join(output_dir, 'epoch_results.csv')
            save_vessel_epoch_results_csv(epoch_results, output_dir)
            evaluate_rank(epoch_results_path=epoch_csv_path, dataset_path=self.get_data_path())

        plot_loss_curves(history, output_dir)
