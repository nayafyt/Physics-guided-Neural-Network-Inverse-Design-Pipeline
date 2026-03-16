from physics.base import PhysicsProblem
import torch
import pandas as pd


class VesselProblem(PhysicsProblem):
    """
    Inverse design: find pressure vessel parameters that minimize min_val.

    NN predicts: SAngle, Nrplies, Stepply, SymLam, Thickpl.
    Physics: soft nearest-neighbor lookup in the dataset (differentiable).
    """

    def __init__(self, alpha=10.0):
        self._inputs = None
        self._targets = None
        self.alpha = alpha

        self.design_params = ['SAngle', 'Nrplies', 'Stepply', 'SymLam', 'Thickpl']
        self.bounds = [(0, 175), (8, 40), (5, 55), (0, 1), (1.0, 2.0)]

        self.X = None
        self.y = None
        self.steps = None
        self.best_design_found = None
        self.best_objective_found = float('inf')

    def get_input_output_dims(self):
        return 1, len(self.design_params)

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        return 'data/pressure_vessel_DS.csv'

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        X = torch.tensor(df[self.design_params].values, dtype=torch.float32)

        obj_values = df["min_val"].values
        obj_tensor = torch.tensor(obj_values, dtype=torch.float32).unsqueeze(1)  # [N, 1]

        obj_min = obj_tensor.min(dim=0).values
        obj_max = obj_tensor.max(dim=0).values
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
        self.y = (obj_tensor - obj_min) / obj_range  # [N, 1], 0 = best, 1 = worst
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

        best_idx = torch.argmin(self.y.squeeze(1)).item()
        self.best_design_found = dict(zip(self.design_params, X[best_idx].tolist()))
        self.best_objective_found = self.y[best_idx].item()

        print(f"\n[VesselProblem] Loaded dataset:")
        print(f"  Alpha: {self.alpha}")
        print(f"  Size: {len(df)}")
        print(f"  Best min_val: {obj_min.item():.6f}")
        print(f"  Best design: {self.best_design_found}")
        print(f"  Parameter steps: {dict(zip(self.design_params, self.steps))}")

        inputs = torch.zeros(1, 1)
        targets = torch.zeros(1, 1)
        self._inputs = inputs
        self._targets = targets
        return inputs, targets

    def forward_physics(self, inputs, predictions):
        X_norm = self.X.clone()
        pred_norm = predictions.clone()
        for i, (lo, hi) in enumerate(self.bounds):
            X_norm[:, i] = (self.X[:, i] - lo) / (hi - lo)
            pred_norm[:, i] = (predictions[:, i] - lo) / (hi - lo)

        diff = pred_norm.unsqueeze(1) - X_norm.unsqueeze(0)
        dist = torch.norm(diff, dim=2)  # [B, N]

        weights = torch.softmax(-self.alpha * dist, dim=1)  # [B, N]
        y_pred = torch.einsum('bn,nk->bk', weights, self.y)  # [B, 1]

        nearest_idx = torch.argmin(dist, dim=1)
        pred_rounded = self.X[nearest_idx]

        min_val = y_pred.min().item()
        if min_val < self.best_objective_found:
            idx = torch.argmin(y_pred.squeeze(1)).item()
            self.best_objective_found = min_val
            self.best_design_found = dict(
                zip(self.design_params, pred_rounded[idx].detach().tolist())
            )

        return y_pred

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
