from physics.base import PhysicsProblem
import torch
import numpy as np
import pandas as pd

OBJECTIVE_MODES = ['min_val', 'multi_objective']


class VesselProblem(PhysicsProblem):
    """
    Inverse design: find pressure vessel parameters that minimize the objective.

    NN predicts: SAngle, Nrplies, Stepply, SymLam, Thickpl.
    Physics: soft nearest-neighbor lookup in the dataset (differentiable).

    Objective modes:
        - 'min_val': pre-computed objective from CSV
        - 'multi_objective': 3 independent objectives (S11/2500, S22/185, Thick*0.12)
    """

    def __init__(self, objective_mode='min_val', alpha_start=10.0, alpha_end=500.0):
        if objective_mode not in OBJECTIVE_MODES:
            raise ValueError(f"objective_mode must be one of {OBJECTIVE_MODES}, got '{objective_mode}'")

        self._inputs = None
        self._targets = None
        self.objective_mode = objective_mode
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha = alpha_start

        self.design_params = ['SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl']
        self.bounds = [(0, 175), (8, 40), (5, 55), (0, 1), (1.0, 2.0)]

        self.X = None
        self.y = None
        self.S11 = None
        self.S22 = None
        self.min_val_min = None
        self.steps = None
        self.best_design_found = None
        self.best_objective_found = float('inf')
        self._last_weights = None
        self._last_nearest_idx = None

    def get_input_output_dims(self):
        # Input = desired objective targets (one per objective)
        obj_dim = 1 if self.objective_mode == 'min_val' else 3
        return obj_dim, len(self.design_params)

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        return 'data/pressure_vessel_DS.csv'

    def _compute_objective(self, df):
        if self.objective_mode == 'min_val':
            return df["min_val"].values

        s11 = df["S11"].values / 2500.0
        s22 = df["S22"].values / 185.0
        thick = df["Thick"].values * 0.12
        return np.column_stack([s11, s22, thick])


    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        X = torch.tensor(df[self.design_params].values, dtype=torch.float32)

        # Store raw S11 and S22 values for constraint checking
        self.S11 = torch.tensor(df["S11"].values, dtype=torch.float32)
        self.S22 = torch.tensor(df["S22"].values, dtype=torch.float32)

        obj_values = self._compute_objective(df)
        obj_tensor = torch.tensor(obj_values, dtype=torch.float32)
        if obj_tensor.dim() == 1:
            obj_tensor = obj_tensor.unsqueeze(1)  # [N, 1]

        obj_min = obj_tensor.min(dim=0).values  # [K]
        obj_max = obj_tensor.max(dim=0).values  # [K]
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0  # avoid division by zero
        # Normalize each objective to [0, 1] so all contribute equally to the loss
        self.y = (obj_tensor - obj_min) / obj_range  # [N, K], 0 = best, 1 = worst per objective
        self.obj_min = obj_min
        self.obj_range = obj_range
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

        # Best design minimizes sum of all normalized objectives (each in [0,1])
        obj_sum = self.y.sum(dim=1)  # [N]
        best_idx = torch.argmin(obj_sum).item()
        self.best_design_found = dict(zip(self.design_params, X[best_idx].tolist()))
        self.best_objective_found = obj_sum[best_idx].item()

        obj_dim = self.y.shape[1]
        print(f"\n[VesselProblem] Loaded dataset:")
        print(f"  Objective mode: {self.objective_mode} | Alpha: {self.alpha_start} -> {self.alpha_end}")
        print(f"  Size: {len(df)} | Objectives: {obj_dim}")
        if obj_dim == 1:
            print(f"  Best objective: {self.obj_min.item():.6f}")
        else:
            labels = ['S11/2500', 'S22/185', 'Thick*0.12']
            mins = self.obj_min.tolist()
            print(f"  Best per objective: {dict(zip(labels, [f'{v:.4f}' for v in mins]))}")
        print(f"  Best design: {self.best_design_found}")
        print(f"  Parameter steps: {dict(zip(self.design_params, self.steps))}")

        # Input = desired objective targets (all zeros = minimize everything)
        # Target = same values, so MSE drives predictions toward these goals
        inputs = torch.zeros(1, obj_dim)
        targets = torch.zeros(1, obj_dim)
        self._inputs = inputs
        self._targets = targets
        return inputs, targets

    def set_epoch(self, epoch, total_epochs):
        """Anneal alpha from alpha_start to alpha_end over training."""
        progress = min(epoch / total_epochs, 1.0)
        self.alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)

    def forward_physics(self, inputs, predictions):
        X_norm = self.X.clone()
        pred_norm = predictions.clone()
        for i, (lo, hi) in enumerate(self.bounds):
            X_norm[:, i] = (self.X[:, i] - lo) / (hi - lo)
            pred_norm[:, i] = (predictions[:, i] - lo) / (hi - lo)

        diff = pred_norm.unsqueeze(1) - X_norm.unsqueeze(0)
        dist = torch.norm(diff, dim=2)  # [B, N]

        weights = torch.softmax(-self.alpha * dist, dim=1)  # [B, N]

        # Soft prediction (for gradients)
        y_soft = torch.einsum('bn,nk->bk', weights, self.y)

        # Hard prediction (actual nearest neighbor objective)
        nearest_idx = torch.argmin(dist, dim=1)  # [B]
        y_hard = self.y[nearest_idx]  # [B, K]

        # Straight-through: use hard value in forward, soft gradients in backward
        y_pred = y_hard + (y_soft - y_soft.detach())

        pred_rounded = self.X[nearest_idx]

        # Cache weights and nearest_idx for use in constraint_loss
        self._last_weights = weights
        self._last_nearest_idx = nearest_idx

        # Best design minimizes sum of all normalized objectives
        obj_sum = y_pred.sum(dim=1)  # [B]
        min_val = obj_sum.min().item()
        if min_val < self.best_objective_found:
            idx = torch.argmin(obj_sum).item()
            self.best_objective_found = min_val
            self.best_design_found = dict(
                zip(self.design_params, pred_rounded[idx].detach().tolist())
            )

        return y_pred

    def constraint_loss(self, predictions):
        """Penalty if designs go outside valid bounds or violate S11/S22 constraints."""
        penalty = 0.0

        # Penalty for bounds violations
        for i, (lo, hi) in enumerate(self.bounds):
            penalty += torch.mean(
                torch.relu(lo - predictions[:, i])**2 +
                torch.relu(predictions[:, i] - hi)**2
            )

        # Penalty for S11 >= 0.6 and S22 >= 0.6 using soft lookup (differentiable)
        if self.S11 is not None and self.S22 is not None and self._last_weights is not None:
            weights = self._last_weights  # [B, N], from forward_physics
            nearest_idx = self._last_nearest_idx  # [B]

            # Normalize S11 and S22 to same scale as objectives
            s11_norm = self.S11 / 2500.0  # [N]
            s22_norm = self.S22 / 185.0   # [N]

            # Soft normalized S11/S22 values (differentiable via weights)
            s11_soft = torch.matmul(weights, s11_norm.unsqueeze(1)).squeeze(1)  # [B]
            s22_soft = torch.matmul(weights, s22_norm.unsqueeze(1)).squeeze(1)  # [B]

            # Hard normalized S11/S22 values (actual nearest neighbor)
            s11_hard = s11_norm[nearest_idx]  # [B]
            s22_hard = s22_norm[nearest_idx]  # [B]

            # Straight-through: hard value in forward, soft gradient in backward
            s11_vals = s11_hard + (s11_soft - s11_soft.detach())
            s22_vals = s22_hard + (s22_soft - s22_soft.detach())

            # Penalize S11/2500 >= 0.6 and S22/185 >= 0.6
            s11_penalty = torch.relu(s11_vals - 0.6)**2
            s22_penalty = torch.relu(s22_vals - 0.6)**2
            penalty += torch.mean(s11_penalty + s22_penalty)

        return penalty

    def get_best_design(self):
        return self.best_design_found

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        import os
        from visualization.plotting import plot_loss_curves, save_vessel_epoch_results_csv, evaluate_rank, plot_multi_objective_curves

        if epoch_results is not None:
            epoch_csv_path = os.path.join(output_dir, 'epoch_results.csv')
            save_vessel_epoch_results_csv(epoch_results, output_dir)
            evaluate_rank(epoch_results_path=epoch_csv_path, dataset_path=self.get_data_path(), objective_mode=self.objective_mode)
            if self.objective_mode == 'multi_objective':
                plot_multi_objective_curves(epoch_csv_path, self.get_data_path(), output_dir)

        plot_loss_curves(history, output_dir)
