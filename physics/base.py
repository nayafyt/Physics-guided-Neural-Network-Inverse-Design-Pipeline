class PhysicsProblem:
    def get_input_output_dims(self):
        raise NotImplementedError

    def load_data(self, path):
        """
        Returns (input_tensor, observed_tensor)
        """
        raise NotImplementedError

    def forward_physics(self, inp, predictions):
        """
        Given:
            inp          : model input tensor (e.g., force, indentation)
            predictions  : predicted material parameters (e.g., E1, E2, E3)
        Returns:
            physics_output : physics-based prediction (e.g., force)
        """
        raise NotImplementedError

    def constraint_loss(self, predictions):
        """
        By default, no constraint penalty.
        Override to enforce physics-based relationships.
        """
        return 0.0
