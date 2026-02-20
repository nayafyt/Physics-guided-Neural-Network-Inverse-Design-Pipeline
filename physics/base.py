class PhysicsProblem:
    """
    Base class for physics-guided problems (continuous and discrete).
    
    Subclasses must implement:
    - get_input_output_dims()
    - get_bounds()
    - get_data_path() - returns path to training data CSV
    - load_data()
    - forward_physics()
    - constraint_loss() [optional]
    - get_output_dir_name() [optional] - custom output directory naming
    - save_results() [optional] - custom result visualization/export
    
    Discreteness is handled through forward_physics() and bounds definition.
    """
    
    def get_input_output_dims(self):
        """
        Returns (input_dim, output_dim)
        - For forward problems: input_dim > 0 (e.g., 2 for indentation)
        - For inverse design: input_dim = 0 (uses latent parameters)
        """
        raise NotImplementedError

    def get_bounds(self):
        """
        Returns list of (min, max) tuples for each output parameter.
        
        Example:
            return [(30, 250), (300, 1200), (30, 300)]  # For 3 outputs
        """
        raise NotImplementedError

    def get_data_path(self):
        """
        Returns path to training data CSV file.
        
        Example:
            return 'data/my_data.csv'
        """
        raise NotImplementedError

    def load_data(self, path):
        """
        Returns (input_tensor, observed_tensor)
        
        For inverse design:
          - input_tensor: dummy value (shape [1, 1])
          - observed_tensor: target objective value
        For forward mapping:
          - input_tensor: actual input data
          - observed_tensor: observed output
        """
        raise NotImplementedError

    def forward_physics(self, inp, predictions):
        """
        Physics model evaluation.
        
        Args:
            inp: [batch_size, input_dim] input tensor
            predictions: [batch_size, output_dim] predicted parameters
            
        Returns:
            [batch_size, 1] physics-based output
        """
        raise NotImplementedError

    def constraint_loss(self, predictions):
        """
        Physics-based constraints (e.g., E2 >= E3).
        
        Returns:
            Scalar penalty (default: 0.0 if no constraints)
        """
        return 0.0

    def get_output_dir_name(self, predictions):
        """
        Generate output directory name based on predictions.
        
        Args:
            predictions: [1, output_dim] final predictions
            
        Returns:
            str: Directory name (will be placed in results/)
            Example: 'vessel_opt_60_24_45_1_1'
        """
        # Default: use rounded integer predictions
        int_vals = predictions.mean(dim=0).cpu().numpy().round().astype(int)
        param_str = '_'.join(str(v) for v in int_vals)
        return f"{self.__class__.__name__}_{param_str}"

    def save_results(self, history, epoch_results, output_dir, predictions, physics_output, inp, obs):
        """
        Save problem-specific results and visualizations.
        
        Args:
            history: Dict with 'total', 'data', 'constraint' loss lists
            epoch_results: List of epoch results (if tracked)
            output_dir: Output directory path
            predictions: Final model predictions
            physics_output: Final physics output
            inp: Input data tensor
            obs: Observed data tensor
        """
        # Default: just plot loss curves
        from visualization.plotting import plot_loss_curves
        plot_loss_curves(history, output_dir)
