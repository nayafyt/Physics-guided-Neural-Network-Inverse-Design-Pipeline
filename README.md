# Physics-Guided Neural Network Inverse Design Pipeline

A machine learning framework for inverse design problems using physics-informed neural networks. The pipeline integrates domain-specific physics constraints directly into neural network training to solve inverse problems.

## Overview

The framework provides a generic, extensible architecture for physics-guided inverse design. Two physics problems are currently implemented:

- **Indentation**: Inverse material characterization — estimates layered material elastic moduli (E1, E2, E3) from experimental force-displacement data
- **Vessel**: Pressure vessel composite layup optimization — finds design parameters (stacking angle, ply count, step, symmetry, thickness) that minimize stress and weight objectives

> **Multi-objective vessel optimization** (S11, S22, Thickness as independent objectives with alpha annealing and S11/S22 constraints) is available on the [`feature/vessel-multi-objective`](../../tree/feature/vessel-multi-objective) branch.

## Project Structure

```
├── main.py                        # Training orchestrator + PGNN architecture
├── physics/
│   ├── base.py                    # Abstract PhysicsProblem base class
│   ├── indentation.py             # Indentation problem implementation
│   └── vessel.py                  # Vessel problem implementation
├── visualization/
│   └── plotting.py                # Plotting and CSV export utilities
├── data/
│   ├── data.csv                   # Indentation experimental data
│   └── pressure_vessel_DS.csv     # Vessel design dataset (52k designs)
├── results/                       # Output directory
├── Dockerfile, docker-compose.yml
└── README.md
```

## Installation

### Local

```bash
git clone <repository-url>
cd Physics-guided-Neural-Network-Inverse-Design-Pipeline
pip install -r requirements.txt
python main.py
```

### Docker

```bash
docker compose up --build
```

## Model Architecture

The PGNN (Physics-Guided Neural Network) is a feedforward network with bounded outputs:

- **Hidden layers**: Configurable depth and width (default: 3 layers × 72 neurons)
- **Activation**: Tanh in hidden layers
- **Output bounding**: Each output is mapped to its physical range via `CustomActivation`:
  ```
  output = min + (max - min) × (tanh(x) + 1) / 2
  ```
- **Normalization**: LayerNorm (batch_size=1) or BatchNorm1d (batch_size>1)
- **Optimizer**: Adam (lr=1e-3) with cosine annealing to 1e-5
- **Gradient clipping**: Max norm 1.0

### Loss Function

```
Total Loss = MSE(targets, physics_output) + λ × constraint_loss
```

## Vessel Problem

### Objective Modes

The vessel problem supports two modes, selected via the `objective_mode` parameter:

- **`min_val`**: Single pre-computed objective from the dataset
- **`multi_objective`**: Three independent normalized objectives:
  - S11/2500 (stress component 1)
  - S22/185 (stress component 2)
  - Thick×0.12 (laminate thickness)

### How It Works

The neural network predicts 5 design parameters and a **soft nearest-neighbor lookup** finds the corresponding objectives from a pre-simulated dataset of 52,272 designs:

1. Normalize predicted design and all dataset entries to [0, 1]
2. Compute Euclidean distance to every dataset design
3. Compute soft weights via temperature-scaled softmax: `w[i] = softmax(-α × dist[i])`
4. **Straight-through estimator**: forward pass uses the hard nearest neighbor value, backward pass uses soft gradients for smooth optimization

### Alpha Annealing

The temperature parameter α anneals linearly from `alpha_start` (default 10) to `alpha_end` (default 500) over training:
- Low α (early): weights spread across many neighbors → exploration
- High α (late): weights concentrate on the single closest neighbor → exploitation

### Constraints

- **Bounds**: Each design parameter is constrained to its physical range via the output activation
- **S11/S22**: Soft penalty for designs with S11/2500 ≥ 0.6 or S22/185 ≥ 0.6, using soft nearest-neighbor weights for proper gradient flow

## Indentation Problem

Estimates layered material elastic moduli (E1, E2, E3) from force-indentation data using a differentiable layered material stiffness model based on beam theory.

- **Input**: Force and indentation measurements
- **Output**: Three elastic moduli (E1, E2, E3) with bounds in MPa
- **Physics**: Computes effective stiffness from predicted moduli and layer geometry
- **Constraint**: Enforces E2 ≥ E3 (stiffness ordering)

## Configuration

All parameters are in `main.py`:

```python
# Problem selection
problem = VesselProblem(objective_mode='multi_objective', alpha_start=10.0, alpha_end=500.0)

# Network architecture
num_hidden_layers = 3      # Try: 2, 3, 4, 5
hidden_dim = 72            # Try: 32, 64, 72, 128

# Training
patience = 250             # Early stopping if no loss improvement
tighten_epochs = 10000     # Maximum training epochs
stable_epochs = 6          # Stop if predictions stable for N epochs
```

## Output

Results are saved to `results/ProblemName_param1_param2_.../`:

- **loss_curves.png**: Total, data, and constraint loss over epochs
- **epoch_results.csv** (vessel): Design parameter and objective evolution
- **multi_objective_curves.png** (vessel, multi_objective mode): S11, S22, Thick convergence plots
- **force_indentation.png** (indentation): Measured vs predicted force curves
- **Pred_E1_E2_E3.csv** (indentation): Predicted force-indentation curve

## Adding a New Physics Problem

Create `physics/your_problem.py` inheriting from `PhysicsProblem`:

```python
from physics.base import PhysicsProblem
import torch

class YourProblem(PhysicsProblem):
    def get_input_output_dims(self):
        return input_dim, output_dim

    def get_bounds(self):
        return [(min1, max1), (min2, max2), ...]

    def get_data_path(self):
        return 'data/your_data.csv'

    def load_data(self, path):
        # Return (input_tensor, target_tensor)
        ...

    def forward_physics(self, inputs, predictions):
        # Differentiable physics model
        ...

    def constraint_loss(self, predictions):
        # Physical constraints (return 0.0 if none)
        return 0.0

    def save_results(self, history, epoch_results, output_dir,
                     predictions, computed_output, inputs, targets):
        from visualization.plotting import plot_loss_curves
        plot_loss_curves(history, output_dir)
```

Then select it in `main.py`:
```python
problem = YourProblem()
```

## Requirements

- Python 3.13+
- PyTorch 2.9+
- NumPy, SciPy, Pandas, Matplotlib
