from physics.base import PhysicsProblem
import torch
import pandas as pd


class VesselProblem(PhysicsProblem):
    """
    Inverse design: find pressure vessel parameters that minimize the objective.

    CSV data: design-space with parameter combos and their objective (min_val).
    NN predicts: SAngle, Nrplies, Stepply, SymLam, Thickpl.
    Physics: soft nearest-neighbor lookup in the dataset (differentiable).
    Loss: drives the NN toward the design with the lowest objective.
    """

    def __init__(self):
        # Cached data for save_results
        self._inputs = None
        self._targets = None

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

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        X = torch.tensor(df[self.design_params].values, dtype=torch.float32)
        min_val = torch.tensor(df["min_val"].values, dtype=torch.float32)

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
        print(f"  Size: {len(df)}")
        print(f"  Best objective (original): {self.min_val_min:.6f}")
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
        alpha = 10.0  # higher = closer to hard nearest-neighbor
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
