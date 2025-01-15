import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class BanditVisualizer:
   def plot_posterior(self, banditA, banditB, iteration, method):
       
       """Create and save posterior distribution plot for both bandits."""
      
       # Calculate Beta distributions
       a_dist = [np.random.beta(banditA.a_prior + banditA.clicks, banditA.b_prior + banditA.views - banditA.clicks) 
                for _ in range(5000)]
       b_dist = [np.random.beta(banditB.a_prior + banditB.clicks, banditB.b_prior + banditB.views - banditB.clicks) 
                for _ in range(5000)]

       plt.figure(figsize=(10, 6))

       sns.kdeplot(data=a_dist, label=f'Button A (CTR: {(banditA.a_prior + banditA.clicks - 1) / (banditA.b_prior + banditA.views - 1):.3f})', 
                   color='#f04b26', linewidth=2)
       sns.kdeplot(data=b_dist, label=f'Button B (CTR: {(banditB.a_prior + banditB.clicks - 1) / (banditB.b_prior + banditB.views - 1):.3f})', 
                   color='#5a18de', linewidth=2)
       
       plt.title(f'{method} Posterior Distributions after {iteration} iterations')
       plt.xlabel('Click-through Rate (CTR)')
       plt.ylabel('Density')
       plt.legend()
       plt.savefig(f'data/figures/posterior_{iteration}.png')
       plt.close()


   def create_grid(self, method, save_path, iterations=[50, 150, 500, 1500, 3000, 5000]):
       
       """Combine saved snapshots into a 2x3 grid."""

       _, axes = plt.subplots(2, 3, figsize=(20, 12))
       
       for idx, iteration in enumerate(iterations):
           img = plt.imread(Path(save_path) / f'posterior_{iteration}.png')
           ax = axes[idx//3, idx%3]
           ax.imshow(img)
           ax.axis('off')
           ax.set_title(f'After {iteration} iterations')
       
       plt.tight_layout()
       plt.savefig(Path(save_path) / f'{method}_posterior_grid.png')
       plt.close()

   
def plot_cumulative_reward(true_ctrs, decisions, alg_names, save_path):
    
    """Plot cumulative rewards over time."""

    rewards = [None] * len(decisions)  # Pre-allocate list
    win_rates = [None] * len(decisions)

    max_length = max(len(array) for array in decisions)  # Find longest array
    
    for i, array in enumerate(decisions):
        rewards[i] = np.cumsum(array)
        win_rates[i] = rewards[i] / (np.arange(len(array)) + 1)

    plt.figure(figsize=(10, 6))
    for i in range(len(win_rates)):
        plt.plot(win_rates[i], label=alg_names[i])
    plt.plot(np.ones(max_length)*np.max(true_ctrs), color="#e62c95", label="Optimal rate")
    plt.axvline(x=600, color='black', linestyle=':', linewidth=0.5) # used for thmompson server
    plt.ylim(0, 0.14)
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Progress: Cumulative Reward Over Time')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xscale('log') # log scale for x axis
    plt.savefig(Path(save_path) / 'cumulative_reward.png')
    plt.close()