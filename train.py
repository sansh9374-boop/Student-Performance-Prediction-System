"""Train all models and save artifacts."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from model import train_models

metrics = train_models(ROOT / "data" / "student_performance.csv", ROOT / "models")
with open(ROOT / "models" / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
