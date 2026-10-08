import json

with open("metrics.json", "r") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

print("Model Accuracy:", accuracy)

if accuracy >= 0.80:
    print("ML QUALITY GATE PASSED")
else:
    print("ML QUALITY GATE FAILED")
    raise SystemExit(1)
