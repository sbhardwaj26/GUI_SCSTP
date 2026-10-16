import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Tells Streamlit to run 'app.py' exactly as if you typed it in the terminal
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())