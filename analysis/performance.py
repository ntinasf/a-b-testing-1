import pandas as pd
import numpy as np
from models.bandits import ThompsonBandit, UCB1Bandit

def analyze_and_save_results(banditA, banditB, original_df, method):
   """Analyze performance and save results to CSV."""
   # Calculate original CTRs
   true_ctr_a = original_df[original_df['button']=='A']['action'].mean()
   true_ctr_b = original_df[original_df['button']=='B']['action'].mean()
   
   # Calculate actual CTRs
   bandit_ctr_a = banditA.clicks / banditA.views
   bandit_ctr_b = banditB.clicks / banditB.views
   
   results = {
       'algorithm': [method],
       'true_ctr_a': [true_ctr_a],
       'true_ctr_b': [true_ctr_b],
       'bandit_ctr_a': [bandit_ctr_a],
       'bandit_ctr_b': [bandit_ctr_b],
       'button_a_clicks': [banditA.clicks],
       'button_a_views': [banditA.views],
       'button_b_clicks': [banditB.clicks],
       'button_b_views': [banditB.views],
   }
   
   # Add algorithm-specific parameters
   if isinstance(banditA, ThompsonBandit):
       results.update({
           'a_alpha': [banditA.a_prior + banditA.clicks],
           'a_beta': [banditA.b_prior + banditA.views - banditA.clicks],
           'b_alpha': [banditB.a_prior + banditB.clicks],
           'b_beta': [banditB.b_prior + banditB.views - banditB.clicks]
       })
   
   return results