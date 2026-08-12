# Deployment Guide

## Streamlit Cloud
1. Create a GitHub repository and upload this project.
2. Install the dependencies from `requirements.txt`.
3. Set the main file to `app.py`.
4. Deploy from the repository.
5. Verify that the dashboard loads and predictions work.

## Render
1. Create a web service connected to the repository.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

A live URL is intentionally not fabricated because deployment requires an external hosting account.
