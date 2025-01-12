# classic_server.py
from flask import Flask, jsonify, request
import logging
from pathlib import Path
import pandas as pd
from models.bandits import ClassicABTest
from analysis.performance import analyze_and_save_results

app = Flask(__name__)

method = "ab_test"
original_df = pd.read_csv("data/click_data.csv")
decisions = []

# Initialize variants
variantA = ClassicABTest("A")
variantB = ClassicABTest("B")
current_assignment = "A"  # Start with A

@app.route("/show")
def show():
    global current_assignment
    decisions.append(0)

    # Alternate between A and B
    if current_assignment == "A":
        variantA.add_view()
        current_assignment = "B"
    else:
        variantB.add_view()
        current_assignment = "A"
           
    logging.info(f"Showing button {current_assignment}")
    return jsonify({"button": current_assignment})

@app.route("/click_button", methods=["POST"])
def click_button():
    result = "OK"
    if request.form["button"] == "A":
        variantA.add_click()
    elif request.form["button"] == "B":
        variantB.add_click()
    else:
        result = "Invalid Input."

    decisions[variantA.views + variantB.views - 1] = 1 
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8888")

    pd.Series(decisions).to_csv('data/ab_simulation_decisions.csv', index=False)
    results = analyze_and_save_results(variantA, variantB, original_df, method)
    pd.DataFrame(results).to_csv(f'data/{method}_results.csv', index=False)

    print(f"\nA: Clicks-{variantA.clicks}, Views-{variantA.views}, CTR-{variantA.clicks / variantA.views:.3f}")
    print(f"\nB: Clicks-{variantB.clicks}, Views-{variantB.views}, CTR-{variantB.clicks / variantB.views:.3f}")