
import csv
from collections import Counter
from pathlib import Path

# Locate the CSV created by our collector
data_path = Path(__file__).parent / "data" / "rps_landmarks.csv"

# Load all examples
with open(data_path, "r", newline="") as file:
    rows = list(csv.DictReader(file))

print("Total examples:", len(rows))

# Count how many examples of each gesture we collected
labels = [row["label"] for row in rows]
print("Examples per gesture:", Counter(labels))

# Count the different recording sessions
sessions = [row["session"] for row in rows]
print("Recording sessions:", len(set(sessions)))

# Check that we have exactly 63 inputs
features = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

print("Number of input features:", len(features))

# Check for missing or invalid numerical values
import math

bad_rows = 0

for row in rows:
    try:
        values = [float(row[name]) for name in features]
        if not all(math.isfinite(v) for v in values):
            bad_rows += 1
    except (ValueError, KeyError):
        bad_rows += 1

print("Invalid rows:", bad_rows)

# Show an example's first few values
if rows:
    print("\nFirst example:")
    print("Label:", rows[0]["label"])
    print("First 9 features:",
          [rows[0][name] for name in features[:9]])
