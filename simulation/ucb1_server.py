import logging
from flask import Flask, jsonify, request
from models.bandits import UCB1Bandit
import pandas as pd

app = Flask(__name__)

method = "UCB1"
banditA = UCB1Bandit("A")
banditB = UCB1Bandit("B") 

decisions = []

@app.route("/show")
def show():

  decisions.append(0)
  n_views = banditA.views + banditB.views

  # Compare samples and select button to show
  if banditA.sample(n_views) > banditB.sample(n_views): 
    button = "A"
    banditA.add_view()
    logging.info(f"Showing button A - Views: {banditA.views}")
  else:
    button = "B"
    banditB.add_view()
    logging.info(f"Showing button B - Views: {banditB.views}")

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

  decisions[banditA.views + banditB.views - 3] = 1

  return jsonify({"result": result})


if __name__ == "__main__":
  app.run(host="127.0.0.1", port="8888")
  pd.Series(decisions).to_csv(f'data/{method}_decisions.csv', index=False)

  print(f"\n A : Clicks-{banditA.clicks}, Views-{banditA.views}, CTR-{banditA.clicks / banditA.views}")
  print(f"\n B : Clicks-{banditB.clicks}, Views-{banditB.views}, CTR-{banditB.clicks / banditB.views}")