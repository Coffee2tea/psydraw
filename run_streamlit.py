import os
import sys
import subprocess

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Check if streamlit is installed
    import streamlit
    print("Streamlit is installed, starting the application...")
except ImportError:
    # Install streamlit if not available
    print("Streamlit not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    print("Streamlit installed successfully.")

# Start the HTP Test page
cmd = [sys.executable, "-m", "streamlit", "run", "src/pages/HTP Test.py"]
print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd) 