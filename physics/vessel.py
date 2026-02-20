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
        self.y = None  # [N] objective values
        self.min_val_min = None
        self.steps = None  # Step size for each parameter (for rounding)

        self.best_design_found = None
        self.best_objective_found = float('inf')

    def get_input_output_dims(self):
        # Dummy input (not used) + 5 design parameters output
        return 1, 5

    def get_bounds(self):
        """Returns bounds for [SAngle, Nrplies, Stepply, SymLam, Thickpl]"""
        return self.bounds

    def get_data_path(self):
        """Returns path to vessel optimization data"""
        return 'data/pressure_vessel_DS.csv'

    def is_discrete(self):
        return True

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        X = torch.tensor(df[self.design_params].values, dtype=torch.float32)
        min_val = torch.tensor(df["min_val"].values, dtype=torch.float32)

        self.min_val_min = min_val.min().item()
        y = min_val - self.min_val_min  # Shift: best design has objective = 0

        self.X = X
        self.y = y

        # Compute step size for each parameter (minimum difference between unique values)
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
        print(f"  Best objective (original): {self.min_val_min:.6f}")
        print(f"  Best design: {self.best_design_found}")
        print(f"  Parameter steps: {dict(zip(self.design_params, self.steps))}")

        inp = torch.zeros(1, 1)  # Dummy input
        obs = torch.zeros(1, 1)  # Target: objective = 0
        self._inp = inp
        self._obs = obs
        return inp, obs

    def forward_physics(self, inp, predictions):
        """
        Soft nearest-neighbor lookup using softmax over distances.
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
        y_pred = torch.sum(weights * self.y.unsqueeze(0), dim=1)  # [B]

        # Track best solution found (with rounding for reporting)
        # Find which dataset point each prediction is closest to
        nearest_idx = torch.argmin(dist, dim=1)
        pred_rounded = self.X[nearest_idx]  # Get actual design from dataset
        
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
    def save_results(self, history, epoch_results, output_dir, predictions, physics_output, inp, obs):
        """Save vessel-specific results: epoch tracking and loss curves."""
        import os
        from visualization.plotting import plot_loss_curves, save_vessel_epoch_results_csv, evaluate_rank
        
        # Save epoch results if available
        if epoch_results is not None:
            epoch_csv_path = os.path.join(output_dir, 'epoch_results.csv')
            save_vessel_epoch_results_csv(epoch_results, output_dir)
            
            # Evaluate rank of predicted design against dataset
            evaluate_rank(epoch_results_path=epoch_csv_path, dataset_path=self.get_data_path())
        
        # Save loss curves
        plot_loss_curves(history, output_dir)