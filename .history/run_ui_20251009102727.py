# Run Streamlit UI Application
import subprocess
import sys

if __name__ == "__main__":
    # Run Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])
