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

## Project Structure

```
├── main.py                     # Generic training orchestrator
├── physics/
│   ├── base.py                 # Abstract PhysicsProblem base class
│   ├── indentation.py          # IndentationProblem implementation
│   └── [your_problem].py       # Add custom physics problems here
├── visualization/
│   └── plotting.py             # Plotting and CSV export utilities
├── data/
│   └── data.csv                # Experimental data
├── results/                  # Output directory with predictions
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

### Training (Indentation Problem)

```bash
python main.py
```

The script will:
1. Load experimental data from `data/data.csv`
2. Train the physics-guided neural network
3. Export predictions to `results/` directory
4. Generate force-indentation plots and loss curves

### Configuration

Edit `main.py` to adjust:

- **Material bounds**: `bounds = [(30, 250), (300, 1200), (30, 300)]` for E1, E2, E3
- **Training parameters**: 
  - `patience=250` (early stopping patience)
  - `tighten_epochs=1500` (total training epochs)
  - `stable_epochs=6` (epochs to verify moduli stability)

### Data Format

`data/data.csv` should contain:
```
ind,force
0.0,0.0
0.01,0.5
...
```
Where `ind` is indentation (mm) and `force` is load (mN).

## Model Architecture

### PGNN (Physics-Guided Neural Network)

- **Input**: Problem-specific measurements (e.g., force + indentation for current problem)
- **Output**: Material/design parameters (e.g., E1, E2, E3 elastic moduli)
- **Hidden layers**: 3 layers × 72 neurons with BatchNorm and Tanh
- **Output activation**: Bounds-constrained via `CustomActivation` (tanh scaling)

### Physics Constraints

- **Forward Model**: Domain-specific differentiable physics (e.g., Hertzian contact + composite mechanics)
- **Constraint Loss**: Enforces physical relationships (e.g., E2 ≥ E3 stiffness ordering)
- **Loss Function**: `Total = MSE(obs, physics_output) + λ × constraint_loss`

## Output

After training, results are saved to `results_E1_E2_E3/`:

- **predictions.csv**: Predicted force-indentation curve
- **force_indentation.png**: Comparison of observed vs. predicted forces
- **loss_curves.png**: Training and constraint loss history

## Extending the Framework

The framework is designed for easy extension to new physics problems. Follow these steps:

### 1. Create a New Physics Problem Class

Create `physics/your_problem.py`:

```python
from physics.base import PhysicsProblem
import torch
import pandas as pd

class YourProblem(PhysicsProblem):
    
    def __init__(self):
        # Initialize problem-specific parameters
        self.param1 = 10.0
        self.param2 = 0.5
    
    def get_input_output_dims(self):
        """Return (input_dim, output_dim) for your problem."""
        return 3, 2  # Example: 3 inputs, 2 outputs
    
    def load_data(self, path):
        """Load and preprocess data. Return (input_tensor, observation_tensor)."""
        df = pd.read_csv(path)
        # Process your data here
        inp = torch.tensor(df[['col1', 'col2', 'col3']].values, dtype=torch.float32)
        obs = torch.tensor(df['target'].values, dtype=torch.float32).unsqueeze(1)
        return inp, obs
    
    def forward_physics(self, inp, predictions):
        """
        Implement your physics model.
        inp: model inputs (e.g., [x1, x2, x3])
        predictions: predicted parameters from neural network (e.g., [p1, p2])
        Returns: physics-based prediction
        """
        x1, x2, x3 = inp[:, 0], inp[:, 1], inp[:, 2]
        p1, p2 = predictions[:, 0], predictions[:, 1]
        
        # Your differentiable physics equations
        output = p1 * x1 + p2 * (x2**2) + x3
        return output.unsqueeze(1)
    
    def constraint_loss(self, predictions):
        """
        Optional: Enforce physical constraints.
        Return scalar loss (0 if no constraints).
        """
        p1, p2 = predictions[:, 0], predictions[:, 1]
        # Example: ensure p2 >= 0.1
        return torch.mean(torch.relu(0.1 - p2)**2)
```

### 2. Update `main.py`

```python
from physics.your_problem import YourProblem  # Import your problem

if __name__ == '__main__':
    problem = YourProblem()  # Instantiate
    bounds = [(lower1, upper1), (lower2, upper2)]  # Define parameter bounds
    
    model, history, inp, obs, predictions, physics_output = train(
        problem=problem,
        bounds=bounds,
        data_path='data/your_data.csv',
        patience=250,
        tighten_epochs=1500,
        stable_epochs=6
    )
```

### 3. Prepare Your Data

Create `data/your_data.csv` with columns matching your `load_data()` implementation.

### 4. Run Training

```bash
python main.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy
- Pandas, Matplotlib

## Training Details

- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: Cosine annealing (T_max=1500, eta_min=1e-5)
- **Gradient clipping**: Max norm 1.0
- **Early stopping**: Triggers on loss plateau or first-decimal stability
