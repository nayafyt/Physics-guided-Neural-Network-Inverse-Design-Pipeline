class PhysicsProblem:
    """
    Base class for physics-guided inverse design problems.

    The pipeline works like this:
        1. The NN predicts unknown design parameters (output_dim values).
        2. forward_physics() plugs those predictions into your physics equations
           and computes a response (e.g. force, displacement, objective value).
        3. The loss is the mismatch between that computed response and the
           observed/experimental data loaded from your CSV.
        4. Gradients flow back through the physics into the NN, so it learns
           which parameter values reproduce the observations.

    Subclasses must implement:
    - get_input_output_dims()
    - get_bounds()
    - get_data_path() 
    - load_data()
    - forward_physics()
    - constraint_loss() [optional]
    - get_output_dir_name() [optional] - custom output directory naming
    - save_results() [optional] - custom result visualization/export
    """

    def get_input_output_dims(self):
        """
        Returns (input_dim, output_dim).

        input_dim : number of input columns from your CSV data.
                    Use 1 if you have a single input column (e.g. frequency).
                    Use N if you have multiple (e.g. frequency + amplitude).
                    For pure inverse design with no real inputs, still use 1
                    (dummy) — the difference is in load_data(), not here.
                    Example: IndentationProblem returns 2 (force + depth).
                    Example: VesselProblem returns 1 (dummy).

        output_dim: number of design parameters the NN predicts.
                    Must equal len(self.design_params) / len(self.bounds).
                    Example: IndentationProblem returns 3 (E1, E2, E3).
                    Example: VesselProblem returns 5 (SAngle, Nrplies, …).

        The pipeline auto-sizes the neural network from these values.
        """
        raise NotImplementedError

    def get_bounds(self):
        """
        Returns list of (min, max) tuples for each output parameter.
        Length must equal output_dim.

        The NN output layer uses tanh scaled to these bounds, so predictions
        are always guaranteed to stay within range during training.

        Example (3 material moduli in MPa):
            return [(30, 250), (300, 1200), (30, 300)]
        Example (5 vessel design params):
            return [(0, 175), (8, 40), (5, 55), (0, 1), (1.0, 2.0)]
        """
        raise NotImplementedError

    def get_data_path(self):
        """
        Returns path to the training data CSV — the observed/experimental data
        that the NN predictions are checked against (NOT the NN output).

        The NN predicts unknown design parameters, forward_physics() turns those
        into a computed response using your equations, and the loss is the
        mismatch between that computed response and the data in this CSV.

        Example:
            return 'data/my_data.csv'
        """
        raise NotImplementedError

    def load_data(self, path):
        """
        Load CSV and return (inputs, targets) as torch tensors.

        This method should also cache any internal state needed later
        (e.g. self._targets, self._input_data for plotting in save_results).

        For a forward problem (like IndentationProblem):
          - Read your CSV columns (e.g. 'depth', 'force')
          - Convert to tensors: input_data [N, 1], targets [N, 1]
          - Cache them: self._input_data = input_data; self._targets = targets
          - Concatenate for NN input: inputs = torch.cat([targets, input_data], dim=1)
          - Return (inputs, targets) where inputs is [N, input_dim]
          - The NN sees all input columns; targets is what the loss compares against

        For an inverse / dataset-lookup (like VesselProblem):
          - Read design params + objective from CSV
          - Store dataset internally (e.g. self.X, self.y)
          - Return dummy tensors: (torch.zeros(1,1), torch.zeros(1,1))
          - forward_physics will use self.X / self.y instead of inputs

        Returns:
            inputs:  [N, input_dim] tensor — fed to the NN as input
            targets: [N, 1] tensor — ground-truth the computed physics output
                     is compared against in the loss function
        """
        raise NotImplementedError

    def forward_physics(self, inputs, predictions):
        """
        Compute the physics response from the NN-predicted parameters.

        This is called every training epoch. The output is compared against
        targets to form the loss: MSE(targets, forward_physics(inputs, predictions)).

        IMPORTANT: use torch operations (not plain numpy) so gradients flow
        back through your equations into the NN for training.

        Args:
            inputs:      [batch, input_dim]  — training inputs from load_data
            predictions: [batch, output_dim] — current NN-predicted design params

        Access individual predicted parameters by column:
            predictions[:, 0]  → 1st design param (entire batch)
            predictions[:, 1]  → 2nd design param
            ...

        Returns:
            [batch, 1] tensor — computed response to compare against targets

        Example (IndentationProblem):
            Extracts indentation depth from inputs[:, 1:2], passes it along
            with predicted E1/E2/E3 into a stiffness calculation, returns
            computed force.

        Example (VesselProblem):
            Ignores inputs entirely. Computes soft nearest-neighbor distance
            between predicted design and the cached dataset (self.X), returns
            interpolated objective value.
        """
        raise NotImplementedError

    def constraint_loss(self, predictions):
        """
        Physics-based constraints (e.g., E2 >= E3).
        
        Returns:
            Scalar penalty (default: 0.0 if no constraints).
        """
        return 0.0

    def get_output_dir_name(self, predictions):
        """
        Generate output directory name from final predictions.
        Override for custom naming (e.g. include objective value).

        Args:
            predictions: [1, output_dim] final predictions

        Returns:
            str: directory name placed under results/
        """
        # Default: ClassName_param1_param2_...
        int_vals = predictions.mean(dim=0).cpu().numpy().round().astype(int)
        param_str = '_'.join(str(v) for v in int_vals)
        return f"{self.__class__.__name__}_{param_str}"

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        """
        Save plots, CSVs, or any problem-specific output after training.

        Override to add custom visualizations. Common things to save:
          - Loss curves (default behavior below)
          - Predicted vs. observed response plot
          - CSV of final predictions and computed outputs
          - Epoch-by-epoch parameter evolution

        Use self._targets / self._input_data (cached in load_data) for plotting.

        Args:
            history:         dict with keys 'total', 'data', 'constraint' (loss per epoch)
            epoch_results:   list of dicts with 'epoch', 'predictions', 'physics_output'
            output_dir:      path to save files into
            predictions:     [batch, output_dim] final NN predictions
            computed_output: [batch, 1] final physics output from those predictions
            inputs:          [batch, input_dim] training inputs
            targets:         [batch, 1] ground-truth targets
        """
        # Default: just plot loss curves
        from visualization.plotting import plot_loss_curves
        plot_loss_curves(history, output_dir)
