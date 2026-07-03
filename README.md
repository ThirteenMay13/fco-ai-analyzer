# FCO AI Analyzer

Production-ready web app to analyze FC Online enhancement logs and return real-time recommendation: DAP or WAIT.

## Core Features

- Parse raw log into sessions and main attempts
- Persistent history accumulation over time
- Undo last session
- Pattern mining, Bayesian, Markov, and ML-based recommendation
- Streamlit dashboard with timeline and exports (HTML, XLSX, PDF)

## Run Locally

1. Create environment and install package:
   - Windows PowerShell:

     .\\.venv\\Scripts\\python.exe -m pip install -e .

2. Start app:

   .\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py

## Deploy to Web

### Option 1: Streamlit Community Cloud, no card required

This is the recommended path if you want a public web app without payment information.

Steps:

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click New app.
5. Select the repo and set the main file to [streamlit_app.py](streamlit_app.py).
6. Deploy.

Important:

- Streamlit Community Cloud is free, but storage is not persistent like a dedicated disk.
- If you want persistent long-term history on the cloud, use the Render path below or move the history/database to an external managed database later.

### Option 2: Render, but this requires a card

Render Blueprint now asks for payment information before deployment.

If you want to stay on Render:

1. Add payment info in Render.
2. Create Blueprint from this repo.
3. Render reads [render.yaml](render.yaml) and creates the service automatically.

Data persistence on Render:

- App uses DATA_DIR (default /var/data on Render).
- SQLite and history logs are saved on mounted disk, so data remains after redeploy/restart.

## Important Paths

- Main app: [streamlit_app.py](streamlit_app.py)
- Persistent history: data/history_log.txt locally, /var/data/history_log.txt on Render
- Persistent database: data/fco_ai_analyzer.sqlite3 locally, /var/data/fco_ai_analyzer.sqlite3 on Render
- Free cloud dependency file: [requirements.txt](requirements.txt)
