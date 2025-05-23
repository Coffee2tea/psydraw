import os
import sys
import streamlit as st

# Add the repository root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up secrets and environment variables
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

if 'OPENAI_BASE_URL' in st.secrets:
    os.environ['OPENAI_BASE_URL'] = st.secrets['OPENAI_BASE_URL']

# Run the Streamlit app directly
try:
    # This is the simplest way to run a Streamlit app directly from the main file
    st.title("PsyDraw - HTP Test")
    
    # Display logo if available
    try:
        from PIL import Image
        logo_path = os.path.join("assets", "logo-3.png")
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=300)
    except Exception as e:
        st.warning(f"Could not load logo: {str(e)}")
    
    # Main page content
    st.write("Loading the application...")
    
    # Add the src directory to the path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Define and run the main app file
    htp_test_path = os.path.join("src", "pages", "HTP Test.py")
    
    # Import using exec to avoid module name issues with spaces
    # This is a workaround for Streamlit Cloud
    with open(htp_test_path, 'r') as f:
        exec(f.read())
    
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