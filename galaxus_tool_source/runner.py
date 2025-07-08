
import sys, shutil, tempfile
from pathlib import Path

def main():
    src = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "app.py"
    tmp = Path(tempfile.mkdtemp())
    dst = tmp / "app.py"
    shutil.copy(src, dst)

    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", str(dst), "--server.headless", "true"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
