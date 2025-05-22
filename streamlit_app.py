import os
import sys
import streamlit as st
import subprocess
from pathlib import Path

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up secrets and environment variables
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Define the path to the HTP Test file
htp_test_path = os.path.join("src", "pages", "HTP Test.py")

# Run the Streamlit app directly
try:
    # This is the simplest way to run a Streamlit app
    if __name__ == "__main__":
        # Add the src directory to the path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
        
        # Import required modules
        import importlib.util
        
        # Create a module name that's Python-friendly (no spaces)
        module_name = "htp_test_page"
        
        # Load the module from the file path
        spec = importlib.util.spec_from_file_location(module_name, htp_test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run the main function
        if hasattr(module, "main"):
            module.main()
        else:
            st.error("Could not find main() function in the HTP Test page")
        
except Exception as e:
    st.error(f"Error starting the application: {str(e)}")
    st.write("Please check the following:")
    st.write("1. The OpenAI API key is correctly set in the Streamlit secrets")
    st.write("2. All dependencies are installed properly")
    st.write("3. The file structure is correct")
    st.stop() 