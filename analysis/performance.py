import pandas as pd
import numpy as np
from models.bandits import ThompsonBandit, UCB1Bandit

def analyze_and_save_results(banditA, banditB, original_df):
   """Analyze performance and save results to CSV."""
   # Calculate original CTRs
   true_ctr_a = original_df[original_df['button']=='A']['action'].mean()
   true_ctr_b = original_df[original_df['button']=='B']['action'].mean()
   optimal_ctr = max(true_ctr_a, true_ctr_b)
   
   # Calculate actual CTRs
   bandit_ctr_a = banditA.clicks / banditA.views
   bandit_ctr_b = banditB.clicks / banditB.views
   actual_ctr = (banditA.clicks + banditB.clicks) / (banditA.views + banditB.views)
   
   # Calculate regret
   total_regret = (optimal_ctr - actual_ctr) * (banditA.views + banditB.views)
   
   results = {
       'algorithm': ['Thompson' if isinstance(banditA, ThompsonBandit) else 'UCB1'],
       'true_ctr_a': [true_ctr_a],
       'true_ctr_b': [true_ctr_b],
       'bandit_ctr_a': [bandit_ctr_a],
       'bandit_ctr_b': [bandit_ctr_b],
       'button_a_clicks': [banditA.clicks],
       'button_a_views': [banditA.views],
       'button_b_clicks': [banditB.clicks],
       'button_b_views': [banditB.views],
       'total_regret': [total_regret]
   }
   
   # Add algorithm-specific parameters
   if isinstance(banditA, ThompsonBandit):
       results.update({
           'a_alpha': [banditA.a_prior +banditA.clicks],
           'a_beta': [banditA.b_prior + banditA.views - banditA.clicks],
           'b_alpha': [banditB.a_prior + banditB.clicks],
           'b_beta': [banditB.b_prior + banditB.views - banditB.clicks]
       })
   
   pd.DataFrame(results).to_csv('data/simulation_results.csv', index=False)
   return results