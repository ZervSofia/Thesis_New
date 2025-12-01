# experiments/generate_synthetic.py

import pandas as pd
import numpy as np
import os

# Settings
n_samples = 1000
n_vars = 20
output_path = "../data/processed/synthetic_data.csv"  # relative to this script

# Make sure folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Generate synthetic data
np.random.seed(42)
data = pd.DataFrame(np.random.randn(n_samples, n_vars),
                    columns=[f"X{i}" for i in range(1, n_vars+1)])

# Save to CSV
data.to_csv(output_path, index=False)
print(f"Synthetic dataset saved to {output_path}")
