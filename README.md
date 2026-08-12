# Student Performance Prediction System

A clean end-to-end machine learning capstone project based on the supplied project brief.

## Scope
- Data collection using a self-created academic dataset
- Missing-value handling and duplicate removal
- Categorical encoding and feature normalization
- Exploratory data analysis
- Performance classification: High / Average / Low
- Score regression
- Streamlit dashboard
- Recommendation engine
- Testing and deployment-ready structure

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib.

## Run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Project structure
- `data/` dataset
- `src/` preprocessing, models and recommendations
- `models/` trained model artifacts and metrics
- `screenshots/` dashboard screenshots
- `docs/` report and presentation
- `tests/` basic test
- `app.py` Streamlit dashboard
- `train.py` training script

## Deployment
The project is deployment-ready for Streamlit Cloud or Render. A live deployment requires a user-owned hosting account/repository, so no fabricated deployment URL is included.
