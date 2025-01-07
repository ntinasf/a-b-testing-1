import logging
from flask import Flask, jsonify, request
import pandas as pd
from models.bandits import ThompsonBandit
from models.visualizations import BanditVisualizer, plot_cumulative_reward
from analysis.performance import analyze_and_save_results

app = Flask(__name__)

original_df = pd.read_csv("data/click_data.csv")
actual_ctr_a = original_df.loc[original_df["button"]=="A"]["action"].mean()
actual_ctr_b = original_df.loc[original_df["button"]=="B"]["action"].mean()

# Initialize bandit instances
banditA = ThompsonBandit("A") 
banditB = ThompsonBandit("B") 

# Initialize the visualization tool
visualizer = BanditVisualizer()
snapshot_points = [40, 100, 300, 700, 1500, 2500]
decisions = []

explorations = 0
exploitations = 0
# Compare samples and select button to sho
@app.route("/show")
def show():
  global explorations, exploitations
  n_views = banditA.views + banditB.views
  decisions.append(0)
  
  sample_a = banditA.sample()
  sample_b = banditB.sample()

  if ThompsonBandit.is_exploring(sample_a, sample_b):
    explorations += 1
  else:
    exploitations += 1

  if sample_a > sample_b: 
    button = "A"
    banditA.add_view()
    logging.info(f"Showing button A - Views: {banditA.views}")
  else:
    button = "B"
    banditB.add_view()
    logging.info(f"Showing button B - Views: {banditB.views}")

  if n_views in snapshot_points:
    visualizer.plot_posterior(banditA, banditB, n_views, 'data/figures')
    
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
  visualizer.create_grid('data/figures')
  results = {
        'A': {'clicks': banditA.clicks, 'views': banditA.views},
        'B': {'clicks': banditB.clicks, 'views': banditB.views}
    }
    
  n_samples = banditA.views + banditB.views
  results = analyze_and_save_results(banditA, banditB, explorations, exploitations, original_df)
  plot_cumulative_reward(true_ctrs=[actual_ctr_a, actual_ctr_b], decisions=decisions, 
                         n_samples=n_samples, save_path="data/figures")

  print(f"\n A : Clicks-{banditA.clicks}, Views-{banditA.views}, CTR-{banditA.clicks / banditA.views}")
  print(f"\n B : Clicks-{banditB.clicks}, Views-{banditB.views}, CTR-{banditB.clicks / banditB.views}")