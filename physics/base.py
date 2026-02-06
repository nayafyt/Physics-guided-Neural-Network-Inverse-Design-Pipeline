class PhysicsProblem:
    """
    Base class for physics-guided problems (both continuous and discrete).
    
    Subclasses must implement:
    - get_input_output_dims()
    - load_data()
    - forward_physics()
    - constraint_loss() [optional]
    - is_discrete() [optional]
    """
    
    def get_input_output_dims(self):
        """
        Returns (input_dim, output_dim)
        - For forward problems: input_dim > 0 (e.g., 2 for indentation)
        - For inverse design: input_dim = 0 (uses latent parameters)
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

    def is_discrete(self):
        """
        Whether outputs are discrete design variables.
        Override to return True for discrete problems.
        """
        return False
