
Galaxus Sell‑out Aggregator  –  Source Pack
==========================================

1. Python 3.9+ installieren (macOS: https://www.python.org/downloads/)
2. Terminal:
   python3 -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\activate.bat
   pip install streamlit pandas openpyxl
3. Entwicklung starten:
   streamlit run app.py
   -> öffnet sich unter http://localhost:8501
4. Eigenständige EXE / macOS-App bauen (optional):
   pip install pyinstaller
   pyinstaller --onefile --clean --add-data app.py:. --hidden-import streamlit.web.runtime.scriptrunner --collect-all streamlit runner.py
   Danach dist/runner (Windows: runner.exe) doppelklicken.
