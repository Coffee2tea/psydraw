import streamlit as st
import os
import tempfile
from simple_analysis import analyze_image

st.set_page_config(
    page_title="Simple HTP Drawing Analyzer",
    page_icon="🖼️",
    layout="wide"
)

st.title("Simple HTP Drawing Analyzer")
st.write("Upload a House-Tree-Person drawing for direct analysis by GPT-4o")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Language selector
language = st.selectbox("Select language", ["English", "Chinese"], index=0)
language_code = "en" if language == "English" else "zh"

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        except TypeError:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Analyze button
    if st.button("Analyze Drawing"):
        with st.spinner("Analyzing the drawing with GPT-4o..."):
            try:
                # Call the analysis function
                analysis = analyze_image(tmp_path, language_code)
                
                # Display results
                with col2:
                    st.subheader("Analysis Results")
                    st.markdown(analysis)
                
                # Add download button for the analysis
                st.download_button(
                    label="Download Analysis",
                    data=analysis,
                    file_name="htp_analysis.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
            
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

st.divider()
st.write("This application uses GPT-4o to analyze House-Tree-Person (HTP) test drawings. The analysis is meant for educational purposes only and should not replace professional psychological assessment.") 