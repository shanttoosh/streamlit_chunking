#!/usr/bin/env python3
"""
Cleanup script for Chunk Optimizer
"""
import os
import shutil
import glob
from pathlib import Path

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*.log",
        "*.pid",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".mypy_cache"
    ]
    
    cleaned_count = 0
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Could not remove {file_path}: {e}")
    
    print(f"✅ Cleaned {cleaned_count} temporary files")

def cleanup_storage():
    """Clean up storage directories"""
    storage_dirs = [
        "storage/chromadb",
        "storage/faiss",
        "storage/cache",
        "data/temp",
        "data/exports"
    ]
    
    for directory in storage_dirs:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Cleaned storage directory: {directory}")
            except Exception as e:
                print(f"⚠️ Could not clean {directory}: {e}")

def cleanup_logs():
    """Clean up log files"""
    log_files = glob.glob("logs/*.log")
    
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"✅ Removed log file: {log_file}")
        except Exception as e:
            print(f"⚠️ Could not remove {log_file}: {e}")

def cleanup_state():
    """Clean up state files"""
    state_files = [
        "current_state.pkl",
        "*.pkl"
    ]
    
    for pattern in state_files:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"✅ Removed state file: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not remove {file_path}: {e}")

def main():
    """Main cleanup function"""
    print("🧹 Cleaning up Chunk Optimizer...")
    
    # Clean temporary files
    print("\n🗑️ Cleaning temporary files...")
    cleanup_temp_files()
    
    # Clean storage
    print("\n💾 Cleaning storage directories...")
    cleanup_storage()
    
    # Clean logs
    print("\n📝 Cleaning log files...")
    cleanup_logs()
    
    # Clean state
    print("\n🔄 Cleaning state files...")
    cleanup_state()
    
    print("\n✅ Cleanup completed successfully!")

if __name__ == "__main__":
    main()
