# A/B Testing with Multi-Armed Bandits

## Project Overview
This project implements and compares different approaches to A/B testing, including classic methods and bandit algorithms, using a simulated web button optimization scenario.

### Background
- Website traffic: ~5000 visitors/week
- Goal: Increase visitor engagement through button design optimization
- **Note**: This is a simplified demonstration project with certain assumptions

### Hypothesis & Metrics
- **Hypothesis**: "A new button design will increase click-through rates compared to the existing design"
- **Overall Evaluation Criterion (OEC)**: Click-through rate (CTR)
- **Baseline**: 7% CTR (based on historical data)
- **Minimum Detectable Effect**: 25% relative improvement (1.75 percentage increase)

## Implementation

### Data Generation
The project simulates user interactions where each visitor:
- Is shown either button A (control) or button B (treatment)
- Generates a binary outcome (1: click, 0: no click)
- We set CTR of button A to be 0.07 (7%) and for button B 0.10 (10%)

### Testing Approaches
1. **Classic A/B Test**
   - Traditional fixed-horizon testing
   - Equal allocation between variants

2. **Thompson Sampling (Two variants)**
   - With random exploration period
   - With informed priors and restricted runtime

3. **UCB1 Algorithm**
   - Upper Confidence Bound approach for comparison purposes

## Replication Guide

### Prerequisites
```bash
# Clone repository
git clone [repository-url]

# Install requirements
pip install -r requirements.txt
```

### Running Experiments
All scripts are located in the [`simulation`](simulation) folder:

1. **Generate Data**
```bash
python data_generator.py  # Creates 20000 rows of click data
```

2. **Classic A/B Test**
```bash
# Terminal 1
python ab_test_server.py

# Terminal 2
python client.py  # Set count<=10000
```

3. **Thompson Sampling with Exploration**
```bash
# Terminal 1
python thompson_server.py  # Configure: method="TS_min_exp", a_prior=1, b_prior=1, minimum_exploration=True

# Terminal 2
python client.py  # Set count<=10000
```

4. **Thompson Sampling with Informed Priors**
```bash
# Terminal 1
python thompson_server.py  # Configure: method="TS_priors", a_prior=6, b_prior=78, minimum_exploration=False

# Terminal 2
python client.py  # Set count<=5000
```

5. **UCB1 Implementation**
```bash
# Terminal 1
python ucb1_server.py

# Terminal 2
python client.py  # Set count<=5000
```

**Note**: Use `Ctrl+C` to stop servers after client completion.


## Results
For detailed analysis and conclusions, see[`notebooks/analysis.ipynb`](analysis/analysis.ipynb).

Key findings:

- Classic A/B testing confirmed Button B's superiority with 95% confidence, requiring ~4,700 samples per variant
- Thompson Sampling implementations both identified the better variant while providing:
  - 15-20% higher cumulative rewards compared to A/B testing
  - More efficient button allocation

- Implementation Comparison:
  - TS with minimum exploration: More thorough exploration, higher confidence (2 weeks)
  - TS with informed priors: Faster decision-making, comparable results (1 week)
  - Both approaches achieved >95% probability of Button B being superior

- Practical Impact: Adaptive methods demonstrated clear advantages in both testing efficiency and user engagement optimization
