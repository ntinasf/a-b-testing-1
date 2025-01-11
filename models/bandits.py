import numpy as np

class ThompsonBandit:
    """Thompson Sampling implementation with Beta prior."""
    def __init__(self, name, a_prior, b_prior, minimum_exploration):
        self.clicks = 0 # Successes for Beta distribution
        self.views = 0 # Total trials 
        self.name = name
        self.a_prior = a_prior
        self.b_prior = b_prior
        self.minimum_exploration = minimum_exploration

    def sample(self):
        # Beta distribution with a=1+successes, b=1+failures
        a = self.a_prior + self.clicks
        b = self.b_prior + self.views - self.clicks
        return np.random.beta(a=a, b=b)
    
    def add_click(self):
        self.clicks += 1

    def add_view(self):
        self.views += 1


class UCB1Bandit:
    """Upper Confidence Bound (UCB1) implementation."""
    def __init__(self, name):
        self.ctr_estimate = 0 # Running CTR estimate
        self.clicks = 0 # Total clicks
        self.views = 1 # Start at 1 to avoid division by zero
        self.name = name

    def sample(self, n_samples):
        return self.ctr_estimate + np.sqrt(3 * np.log(n_samples) / self.views)
    
    def add_click(self):
        self.clicks += 1
        # Update running average
        self.ctr_estimate = ((self.views - 1)*self.ctr_estimate + 1) / self.views

    def add_view(self):
        self.views += 1

    @staticmethod
    def is_exploring(bound_a, bound_b, n_total, min_views):
        return abs(bound_a - bound_b) < np.sqrt(2 * np.log(n_total) / min_views)