import logging
from flask import Flask, jsonify, request
from models.bandits import ThompsonBandit, UCB1Bandit
from models.visualizations import BanditVisualizer

app = Flask(__name__)

# Initialize bandit instances
#banditA = ThompsonBandit("A") # -
#banditB = ThompsonBandit("B") # -

# Alternative UCB1 implementation
banditA = UCB1Bandit("A") # --
banditB = UCB1Bandit("B") # --

# Initialize the visualization tool
visualizer = BanditVisualizer()
snapshot_points = [40, 100, 300, 700, 1500, 2500]

# Compare samples and select button to sho
@app.route("/show")
def show():
  n_views = banditA.views + banditB.views
  if banditA.sample(n_views) > banditB.sample(n_views): # --
  #if banditA.sample() > banditB.sample(): # -
    button = "A"
    banditA.add_view()
    logging.info(f"Showing button A - Views: {banditA.views}")
  else:
    button = "B"
    banditB.add_view()
    logging.info(f"Showing button B - Views: {banditB.views}")

  if n_views in snapshot_points:
    #visualizer.plot_posterior(banditA, banditB, n_views, 'data/figures') # -
    visualizer.plot_ucb_evolution(banditA, banditB, n_views, 'data/figures') # --

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

  return jsonify({"result": result})


if __name__ == "__main__":
  app.run(host="127.0.0.1", port="8888")
  visualizer.create_grid('data/figures')
  print(f"\n A : Clicks-{banditA.clicks}, Views-{banditA.views}, CTR-{banditA.clicks / banditA.views}")
  print(f"\n B : Clicks-{banditB.clicks}, Views-{banditB.views}, CTR-{banditB.clicks / banditB.views}")