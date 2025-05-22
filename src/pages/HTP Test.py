import base64
import os
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import requests
import streamlit as st
from PIL import Image

# Use our custom ChatOpenAI wrapper instead of the original
from src.custom_chat_openai import ChatOpenAI
from src.model_langchain import HTPModel

# Add monkey patch to disable proxies in OpenAI
import openai
# Override any proxy settings that might be configured in the environment or elsewhere
if hasattr(openai, '_client'):
    if hasattr(openai._client, 'proxies'):
        delattr(openai._client, 'proxies')

# Constants
BASE_URL = "https://api.openai.com/v1"
MAX_IMAGE_SIZE = (800, 800)

# Supported languages and their codes
SUPPORTED_LANGUAGES = {
    "English": "en"
}

# Model configurations
TEXT_MODEL = "gpt-4-1106-preview"  # Latest GPT-4 Turbo for text analysis
MULTIMODAL_MODEL = "gpt-4-vision-preview"  # GPT-4V for image analysis

# Example
SAMPLE_IMAGES = {
    "example1": "example/example1.jpg",
    "example2": "example/example2.jpg",
    "example3": "example/example3.jpg",
    "example4": "example/example4.jpg",
}

# Language dictionaries
LANGUAGES = {
    "en": {
        "app_title": "🏡 House-Tree-Person Projective Drawing Test",
        "welcome_message": "Welcome to the House-Tree-Person (HTP) projective drawing test application.",
        "instructions_title": "📋 Test Instructions",
        "instructions": """
            **Please read the following instructions carefully:**

            1. **Drawing Requirements**: On a piece of white paper, use a pencil to draw a picture that includes a **house**, **trees**, and a **person**.
            2. **Be Creative**: Feel free to draw as you like. There are no right or wrong drawings.
            3. **No Aids**: Do not use rulers, erasers, or any drawing aids.
            4. **Take Your Time**: There is no time limit. Take as much time as you need.
            5. **Upload the Drawing**: Once you've completed your drawing, take a clear photo or scan it, and upload it using the sidebar.
            6. **Sample Drawings**: We have prepared sample drawings from publicly available internet data for you to explore. You can find them in the sidebar.
            
            **Note**: All information collected in this test will be kept strictly confidential.
        """,
        "upload_prompt": "👉 Please upload your drawing using the sidebar.",
        "analysis_complete": "✅ **Analysis Complete!** You can download the full report from the sidebar.",
        "analysis_summary": "🔍 Analysis Summary:",
        "initial_analysis": "Initial HTP Drawing Analysis",
        "deeper_analysis": "In-Depth Psychological Analysis",
        "initial_analysis_title": "### Initial HTP Drawing Analysis",
        "deeper_analysis_title": "### Deeper Psychological Interpretation",
        "image_uploaded": "⚠️ Image uploaded. Click **Start Analysis** in the sidebar to proceed.",
        "disclaimer": """
            **Disclaimer**:
            - This test is for reference only and cannot replace professional psychological diagnosis.
            - If you feel uncomfortable or experience emotional fluctuations during the test, please stop immediately and consider seeking help from a professional.
            """,
        "model_settings": "🍓 Model Settings",
        "analysis_settings": "🔧 Analysis Settings",
        "report_language": "Report Language:",
        "upload_drawing": "🖼️ Upload Your Drawing:",
        "start_analysis": "🚀 Start Analysis",
        "reset": "♻️ Reset",
        "download_report": "⬇️ Download Report",
        "download_help": "Download the analysis report as a text file.",
        "uploaded_drawing": "📷 Your Uploaded Drawing",
        "error_no_image": "Please upload an image first.",
        "analyzing_image": "Analyzing image, please wait...",
        "error_analysis": "Error during analysis: ",
        "session_reset": "Session has been reset. You can now upload a new image.",
        "sample_drawings": "📊 Sample Drawings",
        "load_sample": "Load Sample {}",
        "sample_loaded": "Sample {} loaded. Click 'Start Analysis' to analyze.",
        "error_no_api_key": "❌ Internal error: Unable to connect to AI service. Please try again later.",
        "ai_disclaimer": "NOTE: AI-generated content, for reference only. Not a substitute for medical diagnosis.",
    }
}

# Helper function to get text based on current language
def get_text(key):
    return LANGUAGES[st.session_state['language_code']][key]

# Helper functions
def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
def resize_image(image: Image.Image, max_size: tuple = MAX_IMAGE_SIZE) -> Image.Image:
    """Resize image if it exceeds max_size."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size)
    return image

def get_model():
    """Initialize and return the HTP model with current API settings."""
    try:
        # Get API key and base URL from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            return None
            
        # Initialize text model
        text_model = ChatOpenAI(
            model_name=TEXT_MODEL,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=4096,
            cache=False
        )
        
        # Initialize multimodal model
        multimodal_model = ChatOpenAI(
            model_name=MULTIMODAL_MODEL,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=4096,
            cache=False
        )
        
        # Initialize HTP model
        model = HTPModel(
            text_model=text_model,
            multimodal_model=multimodal_model,
            language=st.session_state.language_code,
            use_cache=False
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def analyze_image() -> None:
    """Perform image analysis and update session state."""
    if st.session_state['image_data'] is None:
        st.error(get_text("error_no_image"))
        return

    # Check if API key is set in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(get_text("error_no_api_key"))
        return

    try:
        model = get_model()
        if not model:
            st.error(get_text("error_no_api_key"))
            return

        inputs = {
            "image_path": st.session_state['image_data'],
            "language": st.session_state['language_code']
        }

        with st.spinner(get_text("analyzing_image")):
            response = model.workflow(**inputs)
            st.session_state['analysis_result'] = response
    except Exception as e:
        st.error(f"{get_text('error_analysis')}{str(e)}")

def reset_session() -> None:
    """Reset session state."""
    for key in ['image_data', 'image_display', 'analysis_result', 'current_sample', 'image_source']:
        if key in st.session_state:
            del st.session_state[key]
    st.success(get_text("session_reset"))

def export_report() -> None:
    if st.session_state.get('analysis_result'):
        if st.session_state["analysis_result"]['classification'] is True:
            merge_analysis = st.session_state['analysis_result'].get('merge', '')
            final_report = st.session_state['analysis_result'].get('final', '').replace("<o>", "").replace("</o>", "")
            disclaimer = get_text("ai_disclaimer")
            
            export_data = f"{disclaimer}\n\n"
            export_data += f"=== {get_text('initial_analysis')} ===\n{merge_analysis}\n\n"
            export_data += f"=== {get_text('deeper_analysis')} ===\n{final_report}"
        else:
            signal = st.session_state['analysis_result'].get('fix_signal', '')
            disclaimer = get_text("ai_disclaimer")
            export_data = f"{disclaimer}\n\n{signal}"
            
        st.sidebar.download_button(
            label=get_text("download_report"),
            data=export_data,
            file_name=f"HTP_Report_{st.session_state['language_code']}.txt",
            mime="text/plain",
            help=get_text("download_help")
        )

def img_to_bytes(img_path):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode()

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

def get_asset_path(filename):
    return os.path.join(ASSETS_DIR, filename)

# UI components
def sidebar() -> None:
    """Render sidebar components."""
    with st.sidebar:
        logo_path = get_asset_path("logo-3.png")
        st.markdown(
            f'<img src="data:image/png;base64,{img_to_bytes(logo_path)}" alt="logo" width="100%">',
            unsafe_allow_html=True
        )
    
    # Initialize source tracking if not present
    if 'image_source' not in st.session_state:
        st.session_state['image_source'] = None
        
    st.sidebar.markdown(f"## {get_text('sample_drawings')}")
    col1, col2 = st.sidebar.columns(2)
    for idx, (sample_name, sample_path) in enumerate(SAMPLE_IMAGES.items()):
        col = col1 if idx % 2 == 0 else col2
        with col:
            # Create a unique key for each sample button
            button_key = f"load_sample_{idx}_{st.session_state.get('image_source', 'init')}"
            if st.button(get_text("load_sample").format(idx+1), key=button_key):
                # Clear any previous analysis results
                if 'analysis_result' in st.session_state:
                    del st.session_state['analysis_result']
                
                # Load the sample image
                with open(sample_path, "rb") as f:
                    image = Image.open(f)
                    image = resize_image(image)
                    st.session_state['image_data'] = pil_to_base64(image)
                    st.session_state['image_display'] = image
                    st.session_state['current_sample'] = sample_name
                    # Track that the current image came from a sample
                    st.session_state['image_source'] = 'sample'
                    # Force a rerun to update the UI immediately
                    st.rerun()

    st.sidebar.markdown(f"## {get_text('analysis_settings')}")
    # Always set language to English
    st.session_state['language'] = "English"
    st.session_state['language_code'] = "en"
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader(
        get_text("upload_drawing"),
        type=["jpg", "jpeg", "png"],
        help=get_text("upload_drawing"),
        key=f"file_uploader_{st.session_state.get('image_source', 'init')}"
    )
    
    if uploaded_file:
        # Clear any previous analysis results
        if 'analysis_result' in st.session_state:
            del st.session_state['analysis_result']
            
        # Load the uploaded image
        image = Image.open(uploaded_file)
        image = resize_image(image)
        st.session_state['image_data'] = pil_to_base64(image)
        st.session_state['image_display'] = image
        # Clear current sample when uploading own image
        if 'current_sample' in st.session_state:
            del st.session_state['current_sample']
        # Track that the current image came from an upload
        st.session_state['image_source'] = 'upload'
    
    # Buttons
    st.sidebar.markdown("---")
    if st.sidebar.button(get_text("start_analysis"), key=f"analyze_button_{st.session_state.get('image_source', 'init')}"):
        analyze_image()
    if st.sidebar.button(get_text("reset"), key=f"reset_button_{st.session_state.get('image_source', 'init')}"):
        reset_session()
        # Force a rerun to update the UI immediately
        st.rerun()
    if st.session_state.get('analysis_result'):
        export_report()

def main_content() -> None:
    """Render main content area."""
    st.title(get_text("app_title"))
    st.write(get_text("welcome_message"))

    # Instructions
    with st.expander(get_text('instructions_title'), expanded=True):
        st.markdown(get_text("instructions"))

    # Display Uploaded Image or Placeholder
    if st.session_state.get('image_display'):
        st.image(
            st.session_state['image_display'],
            caption=get_text("uploaded_drawing"),
            use_column_width=True
        )
    else:
        st.info(get_text("upload_prompt"))

    # Display Analysis Results
    if st.session_state.get('analysis_result'):
        st.success(get_text("analysis_complete"))
        
        # Check if the analysis contains a warning about inability to analyze
        analysis_text = st.session_state['analysis_result'].get('final', '')
        unable_to_analyze = any(phrase in analysis_text.lower() for phrase in [
            "unable to properly analyze the image",
            "unable to analyze the image",
            "strongly recommend consulting a professional"
        ])
        
        if unable_to_analyze:
            st.error("""
            ⚠️ **WARNING: ANALYSIS LIMITATIONS DETECTED** ⚠️
            
            The system was unable to properly analyze this image. This may be due to:
            - Image quality or format issues
            - Technical limitations of the system
            - Complex elements requiring professional interpretation
            
            **RECOMMENDATION: Please consult with a professional psychologist for proper assessment.**
            """)
        
        # Hide the Analysis Summary section as requested by the user
        # Only show the deeper analysis sections
        
        # Display initial GPT-4o analysis
        with st.expander(get_text("initial_analysis"), expanded=True):
            st.markdown(get_text("initial_analysis_title"))
            st.write(st.session_state['analysis_result'].get('merge', get_text('error_no_image')))
            
        # Display deeper psychological analysis
        with st.expander(get_text("deeper_analysis"), expanded=True):
            st.markdown(get_text("deeper_analysis_title"))
            st.write(st.session_state['analysis_result'].get('final', get_text('error_no_image')))
            
    elif st.session_state.get('image_data') and not st.session_state.get('analysis_result'):
        st.warning(get_text("image_uploaded"))

    # Footer
    st.markdown("---")
    st.markdown(get_text("disclaimer"))

# Main app
def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="PsyDraw: HTP Test", page_icon="🏡", layout="wide")
        
    # Initialize session state variables if not present
    if 'language' not in st.session_state:
        st.session_state['language'] = 'English'
    if 'language_code' not in st.session_state:
        st.session_state['language_code'] = 'en'
    for key in ['image_data', 'image_display', 'analysis_result', 'image_source']:
        if key not in st.session_state:
            st.session_state[key] = None

    sidebar()
    main_content()

if __name__ == "__main__":
    main()