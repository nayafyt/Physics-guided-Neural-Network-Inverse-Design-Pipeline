# Physics-Guided Neural Network Inverse Design Pipeline

A machine learning framework for inverse design problems using physics-informed neural networks (PINNs). This pipeline solves inverse problems by integrating domain-specific physics constraints directly into neural network training.

## Overview

This project provides a generic, extensible framework for physics-guided inverse design. The current implementation targets the **inverse indentation problem**, estimating layered material elastic moduli (E1, E2, E3) from experimental force-displacement data, but the architecture supports any physics-constrained inverse problem.

## Key Features

- **Physics-Constrained Learning**: Enforces domain-specific relationships via custom loss terms
- **Generic Framework**: Extensible `PhysicsProblem` base class for multiple physics domains
- **Deterministic Setup**: Fixed random seeds for reproducible results
- **Cosine-Annealed Learning Rate**: Balances exploration and exploitation
- **Stability-Based Early Stopping**: Terminates when predictions converge
- **Automated Result Export**: Generates CSVs and visualization plots
- **Modular Architecture**: Easy to swap physics implementations

## Supported Physics Problems

### Current
- **Indentation**: Inverse hyperelastic material characterization via nanoindentation
- **Vessel**: Pressure vessel optimization with discrete design variables


## Project Structure

```
├── main.py                     # Generic training orchestrator
├── physics/
│   ├── base.py                 # Abstract PhysicsProblem base class
│   ├── indentation.py          # IndentationProblem implementation
│   ├── vessel.py               # VesselProblem implementation
│   └── [your_problem].py       # Add custom physics problems here ←
├── visualization/
│   └── plotting.py             # Plotting and CSV export utilities
├── data/
│   └── data.csv                # Experimental data
├── results/                    # Output directory with predictions
├── Dockerfile, docker-compose.yml
└── README.md
```

## Installation

### Local Setup

```bash
git clone <repository-url>
cd Physics-guided-Neural-Network-Inverse-Design-Pipeline
```

### Docker Setup

```bash
docker compose up --build
```

## Usage

### Training

```bash
python main.py
```

The script will:
1. Load data from the problem's data file (specified by `problem.get_data_path()`)
2. Train the physics-guided neural network
3. Export predictions to `results/` directory
4. Generate visualization plots and loss curves

### Configuration

All configuration options are in `main.py` with detailed comments explaining what each parameter does:

- **Problem Selection** (line ~247): Choose between `IndentationProblem`, `VesselProblem`, or your custom problem
- **Network Architecture** (line ~257):
  - `num_hidden_layers`: Number of hidden layers (try 2-5)
  - `hidden_dim`: Width of each layer (try 32, 64, 72, 128)
- **Training Hyperparameters** (line ~263):
  - `patience`: Early stopping patience if loss plateaus (default 250)
  - `tighten_epochs`: Maximum training epochs (default 1500)
  - `stable_epochs`: Consecutive stable epochs before stopping (default 6)

The comments in `main.py` provide tuning guidance for each parameter based on whether your model is underfitting, overfitting, or converging too early.

### Data Format

Each physics problem specifies its own data path via `get_data_path()`. Example for indentation:

`data/data.csv`:
```
ind,force
0.0,0.0
0.01,0.5
0.02,1.2
...
```
Where `ind` is indentation (mm) and `force` is load (mN).

## Model Architecture

### PGNN (Physics-Guided Neural Network)

- **Input**: Problem-specific measurements (e.g., [force, indentation] for indentation problem)
- **Output**: Material/design parameters (e.g., [E1, E2, E3] elastic moduli)
- **Hidden layers**: Configurable (default 3 layers × 72 neurons)
- **Normalization**: LayerNorm for single-sample problems, BatchNorm1d for batch problems
- **Activation**: Tanh in hidden layers, bounded output via `CustomActivation` (tanh scaling to parameter ranges)

### Physics Integration

- **Forward Physics**: Domain-specific differentiable model (e.g., Hertzian contact mechanics for indentation)
- **Constraint Loss**: Enforces physical relationships (e.g., E2 ≥ E3 stiffness ordering)
- **Total Loss**: `Total = MSE(observed, physics_output) + λ × constraint_loss`
- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: Cosine annealing (T_max=tighten_epochs, eta_min=1e-5)
- **Gradient Clipping**: Max norm 1.0 to stabilize training

## Output

After training, results are saved to `results/ProblemName_param1_param2_param3/`:

- **loss_curves.png**: Training/constraint loss over epochs
- **force_indentation.png** (indentation): Measured vs predicted forces
- **epoch_results.csv** (vessel): Design parameter evolution during optimization
- **Pred_E1_E2_E3.csv** (indentation): Final predicted force-indentation curve

## Quick Start: Add Your Own Physics Problem

The framework is modular and extensible. Here's the quickest path to add a new problem:

### Step 1: Create Your Physics Problem Class

Create `physics/your_problem.py`:

```python
from physics.base import PhysicsProblem
import torch
import pandas as pd

class YourProblem(PhysicsProblem):
    def __init__(self):
        # Initialize your problem-specific parameters
        pass
    
    def get_input_output_dims(self):
        """Return (input_dim, output_dim) for your problem."""
        return 3, 2  # Example: 3 inputs → 2 outputs
    
    def get_bounds(self):
        """Return parameter bounds as list of (min, max) tuples."""
        return [(0, 100), (1, 50)]  # Bounds for each output
    
    def get_data_path(self):
        """Return path to your data CSV file."""
        return 'data/your_data.csv'
    
    def load_data(self, path):
        """Load and preprocess data. Return (input_tensor, observation_tensor)."""
        df = pd.read_csv(path)
        inp = torch.tensor(df[['col1', 'col2', 'col3']].values, dtype=torch.float32)
        obs = torch.tensor(df['target'].values, dtype=torch.float32).unsqueeze(1)
        return inp, obs
    
    def forward_physics(self, inp, predictions):
        """Implement your differentiable physics model."""
        x = inp[:, 0]
        p1, p2 = predictions[:, 0], predictions[:, 1]
        output = p1 * x + p2  # Your physics equations here
        return output.unsqueeze(1)
    
    def constraint_loss(self, predictions):
        """Optional: Enforce physical constraints. Return scalar loss."""
        return 0.0  # No constraints if not needed
    
    def save_results(self, history, epoch_results, output_dir, predictions, physics_output, inp, obs):
        """Optional: Custom result saving. Call parent method or implement custom plots."""
        from visualization.plotting import plot_loss_curves
        plot_loss_curves(history, output_dir)
```

### Step 2: Switch Problem in main.py

In `main.py`, around line 247:

```python
from physics.your_problem import YourProblem  # Add this import

if __name__ == '__main__':
    problem = YourProblem()  # ← Change this line
    # problem = IndentationProblem()
    # problem = VesselProblem()
```

### Step 3: Prepare Your Data

Create `data/your_data.csv` with columns matching your `load_data()` method.

### Step 4: Run Training

```bash
python main.py
# or with Docker:
docker compose up --build
```

Results appear in `results/YourProblem_param1_param2_param3/`

## Extending the Framework (Detailed Reference)

For advanced customization beyond the Quick Start, here are all the methods you can override:

### PhysicsProblem Base Class Methods

Every physics problem must inherit from `PhysicsProblem` and implement:

```python
class PhysicsProblem(ABC):
    @abstractmethod
    def get_input_output_dims(self) -> tuple:
        """Return (input_dim, output_dim) for your problem."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> list:
        """Return parameter bounds: [(min1, max1), (min2, max2), ...]"""
        pass
    
    @abstractmethod
    def get_data_path(self) -> str:
        """Return path to data CSV file."""
        pass
    
    @abstractmethod
    def load_data(self, path: str) -> tuple:
        """Load data. Return (input_tensor, observation_tensor)."""
        pass
    
    @abstractmethod
    def forward_physics(self, inp, predictions):
        """Implement physics model. Returns physics-based predictions."""
        pass
    
    @abstractmethod
    def constraint_loss(self, predictions):
        """Optional constraints. Return scalar loss (0 if none)."""
        pass
    
    def get_output_dir_name(self, predictions) -> str:
        """Optional: Custom output directory naming. Default uses parameter values."""
        pass
    
    def save_results(self, history, epoch_results, output_dir, predictions, physics_output, inp, obs):
        """Optional: Custom result visualization and saving."""
        pass
```

### Real-World Example: IndentationProblem

See [physics/indentation.py](physics/indentation.py) for a complete example implementing all methods for inverse material characterization.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy
- Pandas, Matplotlib

## Training Details

### Optimization Strategy

- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: Cosine annealing (T_max=tighten_epochs, eta_min=1e-5) for balanced exploration/exploitation
- **Gradient Clipping**: Max norm 1.0
- **Early Stopping**: Triggered by either:
  - Loss plateau (no improvement for `patience` epochs)
  - Output stability (predictions stable to first decimal for `stable_epochs` epochs)

### Loss Function

```
Total Loss = Data Loss + λ × Constraint Loss

Data Loss = MSE(observations, physics_output)
Constraint Loss = physics_problem.constraint_loss(predictions)
λ = 1.0
```

### Reproducibility

- Deterministic setup with fixed seed (SEED=42)
- Disabled CUDA non-determinism
- Xavier initialization for weights
- Consistent results across runs
