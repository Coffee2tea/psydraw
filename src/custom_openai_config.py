import os
import sys
import openai
import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_clean_openai_client(**kwargs):
    """Create a clean OpenAI client without proxy settings."""
    # Only keep essential parameters
    clean_kwargs = {}
    for key in ['api_key', 'base_url', 'api_version', 'organization']:
        if key in kwargs and kwargs[key] is not None:
            clean_kwargs[key] = kwargs[key]
    
    logger.info(f"Creating clean OpenAI client with params: {list(clean_kwargs.keys())}")
    return openai.OpenAI(**clean_kwargs)

def patch_openai():
    """
    Patch OpenAI client to remove proxy settings that might be causing issues.
    """
    logger.info("Attempting to patch OpenAI configuration...")
    
    # Remove proxy-related environment variables
    for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy']:
        if env_var in os.environ:
            logger.info(f"Removing environment variable: {env_var}")
            os.environ.pop(env_var)
    
    # Try to patch openai module
    try:
        if hasattr(openai, '_client'):
            if hasattr(openai._client, 'proxies'):
                logger.info("Removing proxies from openai._client")
                delattr(openai._client, 'proxies')
                
        # Force reload the openai module
        logger.info("Reloading openai module")
        importlib.reload(openai)
        
        # Monkey patch the OpenAI client initialization
        original_init = openai.OpenAI.__init__
        
        def patched_init(self, *args, **kwargs):
            # Remove proxies if present
            if 'proxies' in kwargs:
                logger.info("Removing 'proxies' from OpenAI initialization")
                del kwargs['proxies']
            return original_init(self, *args, **kwargs)
        
        openai.OpenAI.__init__ = patched_init
        logger.info("Successfully patched OpenAI client initialization")
        
        # Try to patch langchain_openai
        try:
            import langchain_openai.chat_models.base
            
            # Save the original method
            original_validate_env = langchain_openai.chat_models.base.ChatOpenAI.validate_environment
            
            # Create a patched method
            def patched_validate_environment(self, values: dict) -> dict:
                """Patched method to avoid proxy issues."""
                try:
                    client_params = {}
                    for k in ["api_key", "organization", "base_url", "timeout"]:
                        if k in values and values[k] is not None:
                            client_params[k] = values[k]
                    
                    # Create clean client without any proxy settings
                    self.root_client = create_clean_openai_client(**client_params)
                    
                    # Set the remaining values as normal
                    for key, value in values.items():
                        setattr(self, key, value)
                    
                    return values
                except Exception as e:
                    logger.error(f"Error in patched validate_environment: {e}")
                    # Fall back to original method
                    return original_validate_env(self, values)
            
            # Apply the patch
            langchain_openai.chat_models.base.ChatOpenAI.validate_environment = patched_validate_environment
            logger.info("Successfully patched langchain_openai.chat_models.base.ChatOpenAI.validate_environment")
            
        except ImportError:
            logger.error("Could not import langchain_openai.chat_models.base")
        except Exception as e:
            logger.error(f"Error patching langchain_openai: {e}")
        
    except Exception as e:
        logger.error(f"Error patching OpenAI: {e}")
    
    return "OpenAI configuration patched" 