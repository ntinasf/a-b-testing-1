"""
Data Generator for A/B Testing Simulation.
Creates synthetic click-through rate (CTR) data for two different buttons
using the Bernoulli distribution to simulate user behavior.
"""

import numpy as np
import pandas as pd

np.random.seed(42) # For reproducibility

N_SAMPLES = 20000 # Total number of views for both buttons
CTR_A = 0.07 # True CTR for button A (7%)
CTR_B = 0.10 # True CTR for button B (10%)

# Generate binary click data
clicks_a = np.random.binomial(n=1, p=CTR_A, size=N_SAMPLES//2) 
clicks_b = np.random.binomial(n=1, p=CTR_B, size=N_SAMPLES//2)

# Create a DataFrame with button labels and their corresponding click actions
df = pd.DataFrame(
    {"button" : np.repeat(["A", "B"], N_SAMPLES//2),
     "action" : np.concatenate([clicks_a, clicks_b])
     }
)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("data/click_data.csv")

# Calculate the actual CTRs from the generated data
actual_ctr_a = df.loc[df["button"]=="A"]["action"].mean()
actual_ctr_b = df.loc[df["button"]=="B"]["action"].mean()

print(f"Actual ctr for button A : {actual_ctr_a:.3f}")
print(f"Actual ctr for button B : {actual_ctr_b:.3f}")
