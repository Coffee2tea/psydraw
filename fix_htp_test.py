import re

# Path to the file
file_path = 'src/pages/HTP Test.py'

# Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Check if the import statements need fixing
if 'from src.custom_chat_openai import ChatOpenAI' in content:
    # Already has the proper path setup before import
    if not 'sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))' in content:
        # Replace the import lines
        new_content = content.replace(
            "from src.custom_chat_openai import ChatOpenAI\nfrom src.model_langchain import HTPModel",
            """# Fix imports for Streamlit Cloud deployment
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now we can import with the correct path
from src.custom_chat_openai import ChatOpenAI
from src.model_langchain import HTPModel"""
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print("Fixed import statements in HTP Test.py")
    else:
        print("Import statements in HTP Test.py already have the proper path setup")
else:
    print("Could not find the expected import statements in HTP Test.py") 