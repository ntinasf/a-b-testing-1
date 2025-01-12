import logging
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from models.bandits import ThompsonBandit
from models.visualizations import BanditVisualizer, plot_cumulative_reward
from analysis.performance import analyze_and_save_results

app = Flask(__name__)

original_df = pd.read_csv("data/click_data.csv")
actual_ctr_a = original_df.loc[original_df["button"]=="A"]["action"].mean()
actual_ctr_b = original_df.loc[original_df["button"]=="B"]["action"].mean()

# Initialize bandit instances
method= "TS_priors"  #"TS_min_exp" # or "TS_priors"
banditA = ThompsonBandit("A", a_prior=6, b_prior=78, minimum_exploration=False) # 6 78 False
banditB = ThompsonBandit("B", a_prior=1, b_prior=1, minimum_exploration=False) # False

# Initialize the visualization tool
visualizer = BanditVisualizer()
snapshot_points = [50, 150, 400, 700, 1300, 2000]
decisions = []

# Compare samples and select button to sho
@app.route("/show")
def show():
  n_views = banditA.views + banditB.views
  minimum_explore = banditA.minimum_exploration and banditB.minimum_exploration
  decisions.append(0)

  if minimum_explore and n_views < 500:
    sample_a = np.random.random()
    sample_b = np.random.random()
  else:
    sample_a = banditA.sample()
    sample_b = banditB.sample()

  if sample_a > sample_b: 
    button = "A"
    banditA.add_view()
    logging.info(f"Showing button A - Views: {banditA.views}")
  else:
    button = "B"
    banditB.add_view()
    logging.info(f"Showing button B - Views: {banditB.views}")

  if n_views in snapshot_points:
    visualizer.plot_posterior(banditA, banditB, n_views, method ,'data/figures')
    
  return jsonify({"button": button})

# Handle button click and update stats
@app.route("/click_button", methods=["POST"])
def click_button():
  result = "OK"
  if request.form["button"] == "A":
    banditA.add_click()
    logging.info(f"Button A clicked - Clicks: {banditA.clicks}")
  elif request.form["button"] == "B":
    banditB.add_click()
    logging.info(f"Button B clicked - Clicks: {banditB.clicks}")
  else:
    result = "Invalid Input."

  decisions[banditA.views + banditB.views - 1] = 1

  return jsonify({"result": result})


if __name__ == "__main__":
  app.run(host="127.0.0.1", port="8888")
  visualizer.create_grid(method=method, save_path='data/figures')
  
  pd.Series(decisions).to_csv(f'data/{method}_decisions.csv', index=False) 
  
  results = analyze_and_save_results(banditA, banditB, original_df, method)
  pd.DataFrame(results).to_csv(f'data/{method}_results.csv', index=False)

  print(f"\n A : Clicks-{banditA.clicks}, Views-{banditA.views}, CTR-{banditA.clicks / banditA.views:.3f}")
  print(f"\n B : Clicks-{banditB.clicks}, Views-{banditB.views}, CTR-{banditB.clicks / banditB.views:.3f}")