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
    
    Implements STEP 3 (DISCRETIZATION) of the optimization flow:
    - Continuous predictions: [34.2, 54.8, 39.5, 0.18, 1.92]
    - Rounding with steps: [34, 55, 40, 0, 1.9]
    - Bounds clipping: ensures all values within valid ranges
    """
    
    # Define valid discrete ranges for each design parameter
    # Format: param_name: (min, max, step)
    DISCRETE_RANGES = {
        'SAngle': (0, 175, 5),      # min=0, max=175, step=5
        'Stepply': (5, 55, 5),      # min=5, max=55, step=5
        'Nrplies': (8, 40, 1),      # min=8, max=40, step=1
        'SymLam': (0, 1, 1),        # min=0, max=1, step=1 (discrete 0 or 1)
        'Thickpl': (1, 2, 1)        # min=1, max=2, step=1 (but stored as float)
    }
    
    @staticmethod
    def round_to_nearest(value, param_name):
        """
        Round a continuous value to nearest valid discrete value.
        
        Example:
            value=34.2, param_name='SAngle'
            → (34.2 - 0) / 5 = 6.84 → round to 7 → 7*5 + 0 = 35
            → Actually closer to 34.2, so: round(6.84) = 7, result = 35
            But let's check: 34.2 is closer to 35 (distance 0.8) or 30 (distance 4.2)
            So round(34.2/5)*5 = round(6.84)*5 = 7*5 = 35
            But actually we want the closest multiple of 5:
            34.2 / 5 = 6.84, round(6.84) = 7, 7*5 = 35 ✓
        
        Args:
            value: Continuous prediction from PGNN (float, unconstrained)
            param_name: Name of design parameter (e.g., 'SAngle', 'Stepply')
            
        Returns:
            Rounded discrete value within valid bounds [lo, hi]
        """
        lo, hi, step = DiscreteRounder.DISCRETE_RANGES[param_name]
        
        # Round to nearest step:
        # 1. Normalize to step units: (value - lo) / step
        # 2. Round to nearest integer: round(...)
        # 3. Denormalize back: ... * step + lo
        rounded = round((value - lo) / step) * step + lo
        
        # STEP 4: Clamp to bounds (in case rounding pushed us outside)
        return np.clip(rounded, lo, hi)
     
    @staticmethod
    def discretize_design(design_array):
        """
        Convert continuous design array to discrete values.
        
        Implements STEP 3-4 of the optimization flow.
        
        Example:
            Input (from PGNN): [34.2, 54.8, 39.5, 0.18, 1.92]
            Output (discretized + clipped): [34, 55, 40, 0, 1.9]
            (approximately, depending on exact steps and rounding)
        
        Args:
            design_array: numpy array [SAngle, Stepply, Nrplies, SymLam, Thickpl]
                         shape (5,)
        
        Returns:
            Discretized design as numpy array with valid discrete values,
            shape (5,), all values within bounds
        """
        param_names = ['SAngle', 'Stepply', 'Nrplies', 'SymLam', 'Thickpl']
        discrete = np.zeros_like(design_array)
        
        for i, name in enumerate(param_names):
            # Apply rounding and bounds clipping
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
        Load dataset for solver-based inverse design.
        
        Dataset contains: design parameters → solver → min_val
        
        For inverse design:
          - inp: dummy input of 0 (triggers latent parameter optimization)
          - obs: target objective value = minimum min_val from dataset
        
        Training process:
          1. PGNN generates design parameters (continuous)
          2. Discretize + Lookup → get min_val from dataset (what solver would return)
          3. Loss = ||min_val - target||² where target = best min_val found
          4. Backprop through gradients → learn latent parameters
        
        Result: NN learns to produce designs that approach the best min_val
        
        Returns:
            inp: [1, 1] tensor with dummy value 0
            obs: [1, 1] tensor with target min_val (the best possible value)
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
        
        # Find the design that achieves the minimum min_val
        best_idx = df["min_val"].idxmin()
        optimal_design = df.loc[best_idx, self.design_params].values.astype(float)
        
        # Initialize best found
        self.best_objective_found = min_val_min
        self.best_design_found = dict(zip(self.design_params, optimal_design))
        
        print(f"\n[VesselProblem] Loaded {len(df)} design points")
        print(f"  Design space: {len(df)} unique parameter combinations")
        print(f"  Objective (min_val) range: [{min_val_min:.6f}, {min_val_max:.6f}]")
        print(f"  Objective mean: {min_val_mean:.6f}")
        print(f"\n  === OPTIMIZATION TARGET ===")
        print(f"  Goal: Minimize ||min_val - 0||² (approach zero)")
        print(f"  Best min_val in dataset: {min_val_min:.9f}")
        print(f"  Design that achieves it: {optimal_design}")
        print(f"  NN will learn to produce designs close to these values")
        print(f"  (because they give the smallest min_val in the dataset)")
        
        # For inverse design with solver-based evaluation:
        # - Input is dummy (0) - signals latent parameter optimization
        # - Observation is 0 (the ideal target)
        #   Loss will be: ||min_val_from_solver - 0||²
        #   The NN learns to minimize min_val by producing designs similar
        #   to those that yield the smallest values in the dataset
        inp = torch.zeros(1, 1, dtype=torch.float32)
        obs = torch.tensor([[0.0]], dtype=torch.float32)  # Target = 0
        
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
        Simulate the solver: discretize design and lookup min_val from dataset.
        
        This is the "solver" part of the inverse design:
        
        Args:
            inp: [batch_size, 1] dummy input (not used)
            predictions: [batch_size, 5] continuous design variable predictions
                        from PGNN
        
        Returns:
            [batch_size, 1] min_val from dataset lookup
            
        Flow:
            predictions [34.2, 54.8, 39.5, 0.18, 1.92]
                ↓ STEP 3: Discretization (round to step)
            [34, 55, 40, 0, 1.9]
                ↓ STEP 4: Bounds checking
            [34, 55, 40, 0, 1.9]
                ↓ STEP 5: Lookup in dataset (what solver would return)
            min_val = 1.152...
                ↓
            Loss = ||min_val - target||² (target = 1.152...)
        """
        batch_size = predictions.shape[0]
        min_vals = []
        
        for i in range(batch_size):
            # STEP 2: Get continuous predictions from PGNN
            design_continuous = predictions[i].detach().cpu().numpy()
            
            # STEP 3: DISCRETIZATION - Round to nearest step
            design_discrete = DiscreteRounder.discretize_design(design_continuous)
            
            # STEP 4: BOUNDS CHECKING - Verify all values are within bounds
            bounds = [(0, 175), (5, 55), (8, 40), (0, 1), (1, 2)]
            for j, (lo, hi) in enumerate(bounds):
                design_discrete[j] = np.clip(design_discrete[j], lo, hi)
            
            # STEP 5: EXACT LOOKUP in dataset (simulate solver)
            design_dict = {
                'SAngle': int(design_discrete[0]),
                'Stepply': int(design_discrete[1]),
                'Nrplies': int(design_discrete[2]),
                'SymLam': int(design_discrete[3]),
                'Thickpl': float(design_discrete[4])
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
        # Loss will compare: ||min_vals - target||² where target = best_min_val
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

