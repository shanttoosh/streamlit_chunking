# Run Streamlit UI Application
import subprocess
import sys
from src.config.settings import settings

if __name__ == "__main__":
    # Run Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/app.py",
        "--server.port", str(settings.UI_PORT),
        "--server.headless", "true"
    ])
