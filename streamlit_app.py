import os
import sys
import streamlit as st

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up secrets and environment variables
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Run the Streamlit app directly
try:
    # Add the src directory to the path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Import and run the HTP Test main function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "pages"))
    
    # Import the main function from HTP Test
    import importlib.util
    htp_test_path = os.path.join("src", "pages", "HTP Test.py")
    spec = importlib.util.spec_from_file_location("htp_test", htp_test_path)
    htp_test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(htp_test_module)
    
    # Call the main function
    htp_test_module.main()
    
except Exception as e:
    st.error(f"Error starting the application: {str(e)}")
    st.write("Please check the following:")
    st.write("1. The OpenAI API key is correctly set in the Streamlit secrets")
    st.write("2. All dependencies are installed properly")
    st.write("3. The file structure is correct")
    
    # Show more detailed error info
    import traceback
    st.code(traceback.format_exc())
    st.stop() 