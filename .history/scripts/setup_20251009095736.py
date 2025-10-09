#!/usr/bin/env python3
"""
Setup script for Chunk Optimizer
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "storage/chromadb",
        "storage/faiss", 
        "storage/cache",
        "data/temp",
        "data/exports",
        "data/sample",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def download_nltk_data():
    """Download NLTK data"""
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('popular')
        print("✅ NLTK data downloaded successfully")
    except ImportError:
        print("⚠️ NLTK not available, skipping data download")
    except Exception as e:
        print(f"⚠️ Failed to download NLTK data: {e}")

def download_spacy_model():
    """Download spaCy model"""
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✅ spaCy model downloaded successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to download spaCy model, will use fallback")
    except Exception as e:
        print(f"⚠️ Error downloading spaCy model: {e}")

def create_env_file():
    """Create .env file from example"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from .env.example")
        else:
            # Create basic .env file
            env_content = """# Environment Variables
API_HOST=localhost
API_PORT=8000
UI_PORT=8501

# OpenAI Configuration (optional)
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
            with open(".env", "w") as f:
                f.write(env_content)
            print("✅ Created basic .env file")
    else:
        print("ℹ️ .env file already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up Chunk Optimizer...")
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data...")
    download_nltk_data()
    
    # Download spaCy model
    print("\n🧠 Downloading spaCy model...")
    download_spacy_model()
    
    # Create .env file
    print("\n⚙️ Setting up environment...")
    create_env_file()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update .env file with your configuration")
    print("2. Run API: python run_api.py")
    print("3. Run UI: python run_ui.py")
    print("4. Access UI at: http://localhost:8501")
    print("5. Access API docs at: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
