import requests
from pathlib import Path
import pandas as pd

# Set up paths and load data
PROJECT_PATH = Path(__file__).parent.parent
data_path = PROJECT_PATH / "data" / "click_data.csv"

# Read and split data by button type
df = pd.read_csv(data_path)
a = df[df["button"] == "A"]
b = df[df["button"] == "B"]
a = a["action"].values
b = b["action"].values

# Print actual CTRs
print("a.mean:", a.mean())
print("b.mean:", b.mean())


i = 0
j = 0
count = 0
while (i < len(a) and j < len(b)) and count <= 2000: # 2000, 5000, 9400
  # Get button recommendation from server
  r = requests.get("http://localhost:8888/show")
  r = r.json()
  # Simulate user action
  if r["button"] == "A":
    action = a[i]
    i += 1
  else:
    action = b[j]
    j += 1

  # Send click data if action is positive
  if action == 1:
    requests.post(
      "http://localhost:8888/click_button",
      data={"button": r["button"]}
    )

  # Progress tracking
  count += 1
  if count % 50 == 0:
    print(f"Seen {count} buttons, A: {i}, B: {j}")