import argparse
import json
import os
import logging
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use our custom ChatOpenAI wrapper instead of the original
from src.custom_chat_openai import ChatOpenAI

from src.model_langchain import HTPModel

# Attempt to disable proxy settings that might be causing issues
import openai
if hasattr(openai, '_client'):
    if hasattr(openai._client, 'proxies'):
        delattr(openai._client, 'proxies')

# Use only OpenAI models for both text and multimodal
TEXT_MODEL = "gpt-4o"
MULTIMODAL_MODEL = "gpt-4o"

def get_args():
    parser = argparse.ArgumentParser(description="HTP Model")
    parser.add_argument("--image_file", type=str, help="Path to the image")
    parser.add_argument("--save_path", type=str, help="Path to save the result")
    parser.add_argument("--language", type=str, default="zh", help="Language of the analysis report")
    parser.add_argument("--use_cache", action="store_true", help="Enable caching (disabled by default)")
    
    return parser.parse_args()

try:
    logger.info("Loading environment variables")
    load_dotenv()
    config = get_args()

    logger.info(f"Arguments: image_file={config.image_file}, save_path={config.save_path}, language={config.language}")
    assert config.language in ["zh", "en"], "Language should be either 'zh' or 'en'."

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    
    logger.info(f"Using API key (first 4 chars): {api_key[:4]}...")
    logger.info(f"Using base URL: {base_url or 'default OpenAI API'}")

    logger.info("Initializing text model")
    text_model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model_name=TEXT_MODEL,
        temperature=0.2,
    )
    
    logger.info("Initializing multimodal model")
    multimodal_model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model_name=MULTIMODAL_MODEL,
        temperature=0.2,
    )

    logger.info("Initializing HTP model")
    model = HTPModel(
        text_model=text_model,
        multimodal_model=multimodal_model,
        language=config.language,
        use_cache=config.use_cache  # Disabled by default unless --use_cache flag is provided
    )

    logger.info("Running HTP workflow")
    result = model.workflow(
        image_path=config.image_file,
        language=config.language
    )

    # save the result to a file
    logger.info(f"Saving results to {config.save_path}")
    with open(config.save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))
        
    logger.info("Analysis completed successfully")

except Exception as e:
    logger.error(f"Error running HTP analysis: {str(e)}", exc_info=True)
    sys.exit(1)