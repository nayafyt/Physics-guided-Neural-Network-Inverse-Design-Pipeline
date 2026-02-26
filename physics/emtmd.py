from physics.base import PhysicsProblem
import torch
import pandas as pd
import numpy as np
import math


class EMTMDProblem(PhysicsProblem):
    """
    EMTMDminverse design problem.

    How to set up your own problem:
    ───────────────────────────────
    1. Define your design parameters in `design_params` (any number of variables) or pass them as variables on initialization.
    2. Set matching bounds in `bounds` — one (min, max) per parameter.
    3. Point `get_data_path()` to your CSV file.
    4. In `load_data()`, map your CSV columns to inputs/targets.
    5. Implement your physics equations in `compute_emtmd_response()`.
    6. Wire it up in `forward_physics()` so the pipeline can call it.
    """

    def __init__(self):
        # Observed / cached data (populated by load_data)
        self._targets = None
        self._input_data = None

        # Design parameters the NN will predict 
        # List every variable name here. The order must match the
        # columns returned by the NN (predictions[:, 0], [:, 1], …).
        self.design_params = [
            # TODO: replace with your actual parameter names, e.g.:
            # 'ms',        # main structure mass (kg)
            # ... add as many as you need
            'param_1',
            'param_2',
            'param_3',
        ]

        # Bounds: one (min, max) per design parameter 
        # Must be same length as design_params.
        self.bounds = [
            # TODO: replace with real physical bounds
            (0.0, 100.0),   # param_1
            (0.0, 100.0),   # param_2
            (0.0, 100.0),   # param_3
        ]

        # Fixed physical constants (not predicted by the NN) 
        # Put anything the physics equations need that is NOT a design
        # variable here.  Access them in compute_emtmd_response() via self.
        # TODO: set your own constants, e.g.:
        # self.g = 9.81            # gravity (m/s²)

    def get_input_output_dims(self):
        """
        Returns (input_dim, output_dim).

        input_dim : number of input columns from your CSV data.
        output_dim: number of design parameters the NN predicts.
                    Must equal len(self.design_params).
        """
        # TODO: adjust input_dim for your data layout
        input_dim = 1                          
        output_dim = len(self.design_params)   # auto-sized from your param list
        return input_dim, output_dim

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        """
        Path to the training data CSV — the observed/experimental data
        that the NN predictions are checked against (NOT the NN output).

        The NN predicts unknown design parameters, forward_physics() turns those
        into a computed response using your equations, and the loss is the
        mismatch between that computed response and the data in this CSV.
        """
        # TODO: point to your data file
        return 'data/emtmd_data.csv'

    def load_data(self, path):
        """
        Load and prepare training data from CSV.

        Your CSV should contain:
          - columns for each design parameter   (optional, for dataset-lookup)
          - a column for the target/objective    (what the NN tries to match)
          - columns for any input observations   (if input_dim > 1)

        Returns:
            inputs  – tensor [N, input_dim]
            targets – tensor [N, 1]
        """
       
        raise NotImplementedError(
            "Implement load_data() for your EMTMD dataset. "
            "See the examples above (Example A / Example B)."
        )

    def forward_physics(self, inputs, predictions):
        """
        Called by the training loop every epoch.

        Args:
            inputs:      [batch, input_dim]  — the training inputs
            predictions: [batch, output_dim] — current NN-predicted params

        Returns:
            [batch, 1] tensor — computed observable that is compared to targets.

        Access individual predicted parameters with:
            predictions[:, 0]  → first design param (whole batch)
            predictions[:, 1]  → second design param
            ...
        """
        raise NotImplementedError(
            "Implement forward_physics() to call your EMTMD equations."
        )

    def compute_emtmd_response(self, predictions, inputs):
        """
        Core EMTMD physics equations.

        This is where your actual math goes.  Unpack predicted parameters
        by column index, use fixed constants from self.*, and return the
        quantity that the pipeline compares against the target.

        IMPORTANT: use torch operations (not plain Python/numpy) so that
        gradients flow back through the computation for training.

        Args:
            predictions: [batch, output_dim] predicted design parameters
            inputs:      [batch, input_dim]  input data (e.g. frequency)

        Returns:
            [batch, 1] tensor — computed response
        """
        raise NotImplementedError(
            "Implement compute_emtmd_response() with your EMTMD equations."
        )

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        """
        Save EMTMD-specific results and plots.

        Override this to export CSV tables, frequency-response plots, etc.
        The base class default just saves loss curves.
        """
        from visualization.plotting import plot_loss_curves
        plot_loss_curves(history, output_dir)
