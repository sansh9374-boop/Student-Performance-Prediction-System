"""Rule-based academic recommendation engine."""

def generate_recommendations(row: dict) -> list[str]:
    recs = []
    attendance = float(row.get("attendance", 0))
    study_hours = float(row.get("study_hours", 0))
    internal_marks = float(row.get("internal_marks", 0))
    participation = float(row.get("participation", 0))
    assignment_completion = float(row.get("assignment_completion", 0))
    weak_subjects = int(row.get("weak_subjects", 0))

    if attendance < 75:
        recs.append("Improve attendance and maintain at least 75% regular attendance.")
    if study_hours < 3:
        recs.append("Increase study hours gradually to at least 3 hours per day.")
    if internal_marks < 50:
        recs.append("Focus on internal assessments and revise core concepts before tests.")
    if participation < 5:
        recs.append("Attend practice sessions and participate more actively in class.")
    if assignment_completion < 70:
        recs.append("Complete assignments on time and use them for weekly revision.")
    if weak_subjects >= 2:
        recs.append("Focus on weak subjects using targeted practice and doubt-clearing sessions.")
    if not recs:
        recs.append("Maintain the current routine and continue regular revision.")
    return recs
