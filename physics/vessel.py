from physics.base import PhysicsProblem
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Discrete Rounding for Inverse Design
# ---------------------------------------------------------
class DiscreteRounder:
    """
    Rounds continuous predictions to nearest valid discrete values
    based on dataset constraints.
    """
    
    DISCRETE_RANGES = {
        'SAngle': (0, 175, 5),      # min, max, step
        'Stepply': (5, 55, 5),
        'Nrplies': (8, 40, 1),
        'SymLam': (0, 1, 1),
        'Thickpl': (1, 2, 1)
    }
    
    @staticmethod
    def round_to_nearest(value, param_name):
        """Round value to nearest valid discrete value for parameter."""
        lo, hi, step = DiscreteRounder.DISCRETE_RANGES[param_name]
        # Round to nearest step
        rounded = round((value - lo) / step) * step + lo
        # Clamp to bounds
        return np.clip(rounded, lo, hi)
    
    @staticmethod
    def discretize_design(design_array):
        """
        Convert continuous design array to discrete values.
        
        Args:
            design_array: numpy array [SAngle, Stepply, Nrplies, SymLam, Thickpl]
        
        Returns:
            Discretized design as numpy array
        """
        param_names = ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
        discrete = np.zeros_like(design_array)
        for i, name in enumerate(param_names):
            discrete[i] = DiscreteRounder.round_to_nearest(design_array[i], name)
        return discrete


class VesselProblem(PhysicsProblem):
    """
    Inverse design problem: find optimal design variables that minimize min_val
    by direct lookup in the dataset.
    
    No surrogate model—uses exact values from the CSV.
    """

    def __init__(self):
        self._inp = None
        self._obs = None
        self.dataset_df = None
        self.design_params = ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']

    # =========================================================
    # Required PhysicsProblem Interface
    # =========================================================

    def get_input_output_dims(self):
        """
        Returns (input_dim, output_dim) for PGNN.
        
        For inverse design: 
          - input_dim = 0 (uses latent parameters)
          - output_dim = 5 (design variables)
        """
        return 0, 5

    def load_data(self, csv_path):
        """
        Load dataset for lookup-based inverse design.
        
        Returns:
            inp: dummy input [1, 1]
            obs: target min_val (mean of dataset as optimization target)
        """
        df = pd.read_csv(csv_path)
        
        # Clean headers
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\ufeff', '', regex=False)
        
        # Store dataset
        self.dataset_df = df
        
        # Get dataset statistics
        min_val_data = df["min_val"].values
        min_val_mean = min_val_data.mean()
        min_val_min = min_val_data.min()
        min_val_max = min_val_data.max()
        
        print(f"\n[VesselProblem] Loaded {len(df)} design points")
        print(f"  Design space: {len(df)} unique combinations")
        print(f"  min_val range: [{min_val_min:.6f}, {min_val_max:.6f}]")
        print(f"  min_val mean:  {min_val_mean:.6f}")
        print(f"  Best min_val:  {min_val_min:.6f}")
        
        # For inverse design:
        # - Input is dummy (no real input data)
        # - Observation is the best (minimum) min_val from dataset
        inp = torch.zeros(1, 1)
        obs = torch.tensor([[min_val_min]], dtype=torch.float32)
        
        return inp, obs

    def lookup_min_val(self, design_dict):
        """
        Lookup min_val from dataset for a specific design.
        
        Args:
            design_dict: dict with keys ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
        
        Returns:
            min_val if found, None otherwise
        """
        if self.dataset_df is None:
            return None
        
        # Build query conditions
        mask = pd.Series([True] * len(self.dataset_df))
        for param, value in design_dict.items():
            mask &= (self.dataset_df[param] == value)
        
        matches = self.dataset_df[mask]
        if len(matches) > 0:
            return matches.iloc[0]["min_val"]
        else:
            return None

    def forward_physics(self, inp, predictions):
        """
        Physics model: lookup actual min_val from dataset.
        
        Args:
            inp: [batch_size, 1] dummy input
            predictions: [batch_size, 5] continuous predictions
        
        Returns:
            [batch_size, 1] min_val from dataset lookup (or nearest neighbor)
        """
        batch_size = predictions.shape[0]
        min_vals = []
        
        for i in range(batch_size):
            # Discretize the prediction
            design_continuous = predictions[i].detach().cpu().numpy()
            design_discrete = DiscreteRounder.discretize_design(design_continuous)
            
            # Build design dict
            design_dict = {
                'SAngle': int(design_discrete[0]),
                'Stepply': int(design_discrete[1]),
                'Nrplies': int(design_discrete[2]),
                'SymLam': int(design_discrete[3]),
                'Thickpl': int(design_discrete[4])
            }
            
            # Lookup in dataset
            min_val = self.lookup_min_val(design_dict)
            
            # If exact match not found, use nearest neighbor
            if min_val is None:
                # Find nearest design in dataset (L2 distance on normalized features)
                dataset_designs = self.dataset_df[self.design_params].values
                distances = np.linalg.norm(dataset_designs - design_discrete, axis=1)
                nearest_idx = np.argmin(distances)
                min_val = self.dataset_df.iloc[nearest_idx]["min_val"]
            
            min_vals.append(float(min_val))
        
        # Return as tensor
        return torch.tensor(min_vals, dtype=torch.float32).unsqueeze(1)

    def constraint_loss(self, predictions):
        """
        Soft constraints to keep predictions within valid bounds.
        
        Args:
            predictions: [batch_size, 5] design variables
        
        Returns:
            Scalar penalty for out-of-bounds predictions
        """
        penalty = 0.0
        
        bounds = [
            (0, 175),   # SAngle
            (5, 55),    # Stepply
            (8, 40),    # Nrplies
            (0, 1),     # SymLam
            (1, 2)      # Thickpl
        ]
        
        for i, (lo, hi) in enumerate(bounds):
            below = torch.relu(lo - predictions[:, i])
            above = torch.relu(predictions[:, i] - hi)
            penalty += torch.mean(below**2 + above**2)
        
        return penalty
