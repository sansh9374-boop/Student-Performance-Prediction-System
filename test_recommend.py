from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from recommend import generate_recommendations

def test_recommendations():
    row = {
        "attendance": 60, "study_hours": 1.5, "internal_marks": 42,
        "participation": 3, "assignment_completion": 50, "weak_subjects": 3
    }
    recs = generate_recommendations(row)
    assert len(recs) >= 5
