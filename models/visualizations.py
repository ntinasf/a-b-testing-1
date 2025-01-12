import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class BanditVisualizer:
   def plot_posterior(self, banditA, banditB, iteration, method, save_path):
       
       """Create and save posterior distribution plot for both bandits."""
      
       # Calculate Beta distributions
       a_dist = [np.random.beta(banditA.a_prior + banditA.clicks, banditA.b_prior + banditA.views - banditA.clicks) 
                for _ in range(5000)]
       b_dist = [np.random.beta(banditB.a_prior + banditB.clicks, banditB.b_prior + banditB.views - banditB.clicks) 
                for _ in range(5000)]

       # Create plot
       plt.figure(figsize=(10, 6))
       sns.kdeplot(data=a_dist, label=f'Button A (CTR: {(banditA.a_prior + banditA.clicks - 1) / (banditA.b_prior + banditA.views - 1):.3f})', 
                   color='#f04b26', linewidth=2)
       sns.kdeplot(data=b_dist, label=f'Button B (CTR: {(banditB.a_prior + banditB.clicks - 1) / (banditB.b_prior + banditB.views - 1):.3f})', 
                   color='#5a18de', linewidth=2)
       
       plt.title(f'{method} Posterior Distributions after {iteration} iterations')
       plt.xlabel('Click-through Rate (CTR)')
       plt.ylabel('Density')
       plt.legend()
       
       # Save plot
       plt.savefig(Path(save_path) / f'posterior_{iteration}.png')
       plt.close()

   def plot_ucb_bounds(self, banditA, banditB, iteration, save_path):
        
        """Create and save UCB1 bounds plot for both bandits."""
        plt.figure(figsize=(10, 6))
   
        # Calculate total samples for UCB term
        n_samples = banditA.views + banditB.views
   
        # Calculate bounds for both bandits
        a_estimate = banditA.ctr_estimate
        a_ucb = np.sqrt(2 * np.log(n_samples) / banditA.views)
   
        b_estimate = banditB.ctr_estimate
        b_ucb = np.sqrt(2 * np.log(n_samples) / banditB.views)
   
        # Plot estimates and bounds
        plt.errorbar(x=['Button A'], y=[a_estimate], 
               yerr=[a_ucb], capsize=10, color='#a2e62c', 
               fmt='o', markersize=10, 
               label=f'A (CTR: {a_estimate:.3f} ± {a_ucb:.3f})')
   
        plt.errorbar(x=['Button B'], y=[b_estimate], 
               yerr=[b_ucb], capsize=10, color='#e66d2c', 
               fmt='o', markersize=10,
               label=f'B (CTR: {b_estimate:.3f} ± {b_ucb:.3f})')
   
        plt.title(f'UCB1 Estimates and Bounds after {iteration} iterations')
        plt.ylabel('Click-through Rate (CTR)')
        plt.ylim(0, max(a_estimate + a_ucb, b_estimate + b_ucb) * 1.1)
        plt.legend()
   
        # Save plot
        plt.savefig(Path(save_path) / f'ucb_bounds_{iteration}.png')
        plt.close()   

   def create_grid(self, save_path, iterations=[50, 150, 400, 700, 1300, 2000], method):
       
       """Combine saved snapshots into a 2x3 grid."""
       _, axes = plt.subplots(2, 3, figsize=(20, 12))
       
       for idx, iteration in enumerate(iterations):
           img = plt.imread(Path(save_path) / f'posterior_{iteration}.png') # -
           #img = plt.imread(Path(save_path) / f'ucb_evolution_{iteration}.png') # --
           ax = axes[idx//3, idx%3]
           ax.imshow(img)
           ax.axis('off')
           ax.set_title(f'After {iteration} iterations')
       
       plt.tight_layout()
       plt.savefig(Path(save_path) / f'{method} posterior_grid.png') # -
       #plt.savefig(Path(save_path) / 'UCB1_grid.png') # --
       plt.close()

   def plot_ucb_evolution(self, banditA, banditB, iteration, save_path):
    """Plot UCB1 estimates and bounds as a time series."""
    plt.figure(figsize=(10, 6))
    
    # Generate x-axis points
    x_points = np.linspace(0, iteration, 100)
    
    # Calculate estimates and bounds for different points in time
    def get_bounds(bandit, n_total):
        estimate = bandit.ctr_estimate
        ucb = np.sqrt(2 * np.log(n_total) / bandit.views)
        return estimate - ucb, estimate + ucb
    
    # Plot estimates with shaded confidence regions
    n_total = banditA.views + banditB.views
    a_lower, a_upper = get_bounds(banditA, n_total)
    b_lower, b_upper = get_bounds(banditB, n_total)
    
    plt.plot([0, iteration], [banditA.ctr_estimate]*2, 'b-', label='Button A Estimate')
    plt.fill_between([0, iteration], 
                    [a_lower]*2, 
                    [a_upper]*2, 
                    alpha=0.2, 
                    color='#a2e62c')
    
    plt.plot([0, iteration], [banditB.ctr_estimate]*2, 'r-', label='Button B Estimate')
    plt.fill_between([0, iteration], 
                    [b_lower]*2, 
                    [b_upper]*2, 
                    alpha=0.2, 
                    color='#e66d2c')
    
    plt.title(f'UCB1 Estimates and Bounds after {iteration} iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Click-through Rate (CTR)')
    plt.legend()
    
    plt.savefig(Path(save_path) / f'ucb_evolution_{iteration}.png')
    plt.close()


def plot_cumulative_reward(true_ctrs, decisions, alg_names, n_samples, save_path):
    """Plot cumulative rewards over time."""

    rewards = []
    win_rates = []
    for i, array in enumerate(decisions):
        rewards[i] = np.cumsum(decisions)
        win_rates[i] = rewards / (np.arange(len(array)) + 1)

    plt.figure(figsize=(10, 6))
    for i in len(win_rates):
        plt.plot(win_rates[i], label=alg_names[i])
    plt.plot(np.ones(n_samples)*np.max(true_ctrs), color="#e62c95", label="Optimal rate")
    plt.axvline(x=500, color='black', linestyle=':', linewidth=0.5) # used for thmompson server
    
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Progress: Cumulative Reward Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add log scale option
    plt.xscale('log')
    

    plt.savefig(Path(save_path) / 'cumulative_reward.png')
    plt.close()