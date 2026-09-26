@echo off
REM Sets up a virtual environment and runs the full benchmark.
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
pytest -q
python scripts\run_experiments.py
python scripts\build_report.py
pause
