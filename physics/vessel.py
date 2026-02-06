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
    based on design parameter constraints.
    
    This ensures the design variables take only valid discrete values
    from the dataset design space.
    """
    
    # Define valid discrete ranges for each design parameter
    DISCRETE_RANGES = {
        'SAngle': (0, 175, 5),      # min, max, step
        'Stepply': (5, 55, 5),
        'Nrplies': (8, 40, 1),
        'SymLam': (0, 1, 1),
        'Thickpl': (1, 2, 1)
    }
    
    @staticmethod
    def round_to_nearest(value, param_name):
        """
        Round a continuous value to nearest valid discrete value.
        
        Args:
            value: Continuous prediction from PGNN
            param_name: Name of design parameter
            
        Returns:
            Rounded discrete value within valid bounds
        """
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
            Discretized design as numpy array with valid discrete values
        """
        param_names = ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
        discrete = np.zeros_like(design_array)
        for i, name in enumerate(param_names):
            discrete[i] = DiscreteRounder.round_to_nearest(design_array[i], name)
        return discrete


class VesselProblem(PhysicsProblem):
    """
    Inverse design problem for pressure vessel optimization.
    
    Find optimal design variables [SAngle, Stepply, Nrplies, SymLam, Thickpl]
    that minimize the objective (min_val) by direct lookup in the dataset.
    
    Key differences from forward problems:
    - No continuous physics model, only discrete dataset lookup
    - Input is dummy (0) - optimization is inverse/generative
    - Output is 5 design variables 
    - forward_physics performs dataset lookup for evaluation
    """

    def __init__(self):
        self._inp = None
        self._obs = None
        self.dataset_df = None
        self.design_params = ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
        self.best_design_found = None
        self.best_objective_found = float('inf')

    def get_input_output_dims(self):
        """
        Returns (input_dim, output_dim) for PGNN.
        
        For inverse design of vessel: 
          - input_dim = 0 (uses latent parameters, no data input)
          - output_dim = 5 (design variables: SAngle, Stepply, Nrplies, SymLam, Thickpl)
        """
        return 0, 5

    def is_discrete(self):
        """
        Indicates that outputs are discrete design variables,
        not continuous physical parameters.
        """
        return True

    def load_data(self, csv_path):
        """
        Load dataset for lookup-based inverse design.
        
        For inverse design:
          - inp: dummy input of 0 (no real input data needed)
          - obs: target objective value (best/minimum from dataset)
        
        Returns:
            inp: [1, 1] tensor with dummy value 0
            obs: [1, 1] tensor with best objective value from dataset
        """
        df = pd.read_csv(csv_path)
        
        # Clean headers
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\ufeff', '', regex=False)
        
        # Store dataset for later lookup
        self.dataset_df = df
        
        # Get dataset statistics
        min_val_data = df["min_val"].values
        min_val_min = min_val_data.min()
        min_val_max = min_val_data.max()
        min_val_mean = min_val_data.mean()
        
        # Initialize best found
        self.best_objective_found = min_val_min
        
        print(f"\n[VesselProblem] Loaded {len(df)} design points")
        print(f"  Design space: {len(df)} unique parameter combinations")
        print(f"  Objective (min_val) range: [{min_val_min:.6f}, {min_val_max:.6f}]")
        print(f"  Objective mean: {min_val_mean:.6f}")
        print(f"  Target (best objective): {min_val_min:.6f}")
        
        # For inverse design:
        # - Input is dummy (0) - signals that we're optimizing, not predicting from data
        # - Observation is the best objective value (target for minimization)
        inp = torch.zeros(1, 1, dtype=torch.float32)
        obs = torch.tensor([[min_val_min]], dtype=torch.float32)
        
        self._inp = inp
        self._obs = obs
        
        return inp, obs

    def lookup_min_val(self, design_dict):
        """
        Lookup min_val (objective) from dataset for a specific design.
        
        Args:
            design_dict: dict with keys ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
                        with integer discrete values
        
        Returns:
            min_val (float) if exact design found, None otherwise
        """
        if self.dataset_df is None:
            return None
        
        # Build query mask for exact match
        mask = pd.Series([True] * len(self.dataset_df))
        for param, value in design_dict.items():
            mask &= (self.dataset_df[param] == value)
        
        matches = self.dataset_df[mask]
        if len(matches) > 0:
            return matches.iloc[0]["min_val"]
        else:
            return None

    def find_nearest_design(self, design_array):
        """
        Find nearest design in dataset using L2 distance.
        Used when exact discretized design is not in dataset.
        
        Args:
            design_array: [5,] numpy array with discretized values
            
        Returns:
            min_val from nearest design in dataset
        """
        if self.dataset_df is None:
            return None
        
        dataset_designs = self.dataset_df[self.design_params].values
        distances = np.linalg.norm(dataset_designs - design_array, axis=1)
        nearest_idx = np.argmin(distances)
        return self.dataset_df.iloc[nearest_idx]["min_val"]

    def forward_physics(self, inp, predictions):
        """
        Physics evaluation for vessel inverse design: dataset lookup.
        
        Process:
          1. Take continuous predictions from PGNN
          2. Discretize to valid design parameter values
          3. Look up objective (min_val) in dataset
          4. Use nearest neighbor if exact design not found
        
        Args:
            inp: [batch_size, 1] dummy input (not used, always 0)
            predictions: [batch_size, 5] continuous design variable predictions
        
        Returns:
            [batch_size, 1] objective values (min_val) from dataset
        """
        batch_size = predictions.shape[0]
        min_vals = []
        
        for i in range(batch_size):
            # Get continuous prediction
            design_continuous = predictions[i].detach().cpu().numpy()
            
            # Discretize to valid design parameter values
            design_discrete = DiscreteRounder.discretize_design(design_continuous)
            
            # Build design dictionary for lookup
            design_dict = {
                'SAngle': int(design_discrete[0]),
                'Stepply': int(design_discrete[1]),
                'Nrplies': int(design_discrete[2]),
                'SymLam': int(design_discrete[3]),
                'Thickpl': int(design_discrete[4])
            }
            
            # Try exact lookup in dataset
            min_val = self.lookup_min_val(design_dict)
            
            # If exact design not found, use nearest neighbor
            if min_val is None:
                min_val = self.find_nearest_design(design_discrete)
            
            # Track best design found during training
            if min_val is not None and min_val < self.best_objective_found:
                self.best_objective_found = min_val
                self.best_design_found = design_dict.copy()
            
            min_vals.append(float(min_val) if min_val is not None else float('inf'))
        
        # Return as [batch_size, 1] tensor
        return torch.tensor(min_vals, dtype=torch.float32).unsqueeze(1)

    def constraint_loss(self, predictions):
        """
        Soft constraint penalty to keep predictions within valid bounds.
        
        Encourages PGNN to stay within the discrete parameter ranges,
        reducing the number of out-of-bounds lookups.
        
        Args:
            predictions: [batch_size, 5] continuous design variables
        
        Returns:
            Scalar penalty for out-of-bounds violations
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
            # Penalize predictions below or above bounds
            below = torch.relu(lo - predictions[:, i])
            above = torch.relu(predictions[:, i] - hi)
            penalty += torch.mean(below**2 + above**2)
        
        return penalty

    def get_best_design(self):
        """
        Return best design found during training.
        
        Returns:
            dict with keys ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
            or None if no valid design found
        """
        return self.best_design_found

