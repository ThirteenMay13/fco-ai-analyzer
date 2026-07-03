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

## Deploy to Web (Render)

This repository is already configured for Render with Docker and persistent disk:

- [render.yaml](render.yaml)
- [Dockerfile](Dockerfile)
- [.streamlit/config.toml](.streamlit/config.toml)

Steps:

1. Push this folder to a GitHub repository.
2. In Render, click New + and choose Blueprint.
3. Select your GitHub repository.
4. Render reads [render.yaml](render.yaml) and creates web service automatically.
5. Wait for first deploy, then open the web URL.

Data persistence:

- App uses DATA_DIR (default /var/data on Render).
- SQLite and history logs are saved on mounted disk, so data remains after redeploy/restart.

## Important Paths

- Main app: [streamlit_app.py](streamlit_app.py)
- Persistent history: data/history_log.txt locally, /var/data/history_log.txt on Render
- Persistent database: data/fco_ai_analyzer.sqlite3 locally, /var/data/fco_ai_analyzer.sqlite3 on Render
