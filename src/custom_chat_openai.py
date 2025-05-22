from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from typing import Any, Dict, Iterator, List, Optional, Union
import openai
import logging
import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a custom ChatGeneration that includes the generations attribute
class CustomChatGeneration(ChatGeneration):
    """Custom ChatGeneration class that adds the generations attribute"""
    
    @property
    def generations(self):
        """Return self as the generations attribute"""
        return [self]
    
    @property
    def llm_output(self):
        """Return an empty dict for the llm_output attribute"""
        return {}

class ChatOpenAI(BaseChatModel):
    """
    Custom ChatOpenAI implementation optimized for OpenAI models.
    """
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    
    def __init__(self, *args, **kwargs):
        # Check if model name is a Claude model and replace with GPT equivalent
        if 'model_name' in kwargs and 'claude' in kwargs['model_name'].lower():
            logger.warning(f"Claude model detected: {kwargs['model_name']}. Replacing with gpt-4o.")
            kwargs['model_name'] = "gpt-4o"
        
        # Update older model names to their current versions
        if 'model_name' in kwargs:
            model_mapping = {
                "gpt-4": "gpt-4o",
                "gpt-4-turbo": "gpt-4o",
                "gpt-4-turbo-preview": "gpt-4o",
                "gpt-4-1106-preview": "gpt-4o",
                "gpt-4-vision-preview": "gpt-4o",
                "gpt-3.5-turbo": "gpt-4o",
                "gpt-3.5-turbo-16k": "gpt-4o"
            }
            if kwargs['model_name'] in model_mapping:
                logger.info(f"Updating model name from {kwargs['model_name']} to {model_mapping[kwargs['model_name']]}")
                kwargs['model_name'] = model_mapping[kwargs['model_name']]
        
        super().__init__(*args, **kwargs)
        
    @property
    def _llm_type(self) -> str:
        return "custom-chat-openai"
    
    def _get_api_url(self):
        """Get the correct API URL for the chat completions endpoint."""
        if self.base_url:
            # If base URL is provided, ensure it has the correct endpoint
            if self.base_url.endswith("/v1"):
                return f"{self.base_url}/chat/completions"
            elif self.base_url.endswith("/v1/"):
                return f"{self.base_url}chat/completions"
            elif not self.base_url.endswith("/chat/completions"):
                return f"{self.base_url}/v1/chat/completions"
            else:
                return self.base_url
        else:
            return "https://api.openai.com/v1/chat/completions"
    
    def _make_direct_api_call(self, messages, model, temperature=0.7, stop=None):
        """Make a direct API call to OpenAI's chat completions endpoint."""
        try:
            # Create a session with retries
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Always use the correct API endpoint for chat completions
            api_url = "https://api.openai.com/v1/chat/completions"
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Check if any message has image content to log it
            has_image = False
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            has_image = True
                            logger.info("Detected image in message payload")
            
            # For multimodal messages, make sure the model can handle images
            if has_image:
                # Make sure we're using the right model variant
                model_to_use = model
                if "gpt-4" in model and "vision" not in model and "gpt-4o" not in model:
                    # Always use GPT-4o for vision capabilities as it handles this natively
                    model_to_use = "gpt-4o"
                    logger.info(f"Upgrading model to {model_to_use} for image analysis")
                
                # Prepare request body with the upgraded model
                data = {
                    "model": model_to_use,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 4096
                }
                
                # Log detailed request info
                logger.info(f"Making API call for image analysis with model {model_to_use}")
                logger.info(f"API URL: {api_url}")
                logger.info("Message content contains image data")
            else:
                # Normal text-only request
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 4096
                }
                logger.info(f"Making API call for text-only with model {model} to {api_url}")
            
            if stop:
                data["stop"] = stop
            
            # Make the request
            logger.info("Sending API request...")
            response = session.post(api_url, headers=headers, json=data)
            # Log response status code
            logger.info(f"Response status code: {response.status_code}")
            
            # If there's an error, try to log the error message
            if response.status_code != 200:
                try:
                    error_json = response.json()
                    logger.error(f"API error: {error_json}")
                except:
                    logger.error(f"API error: {response.text[:500]}")
                    
            response.raise_for_status()
            
            # Process the response
            result = response.json()
            logger.info("API call completed successfully")
            
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error in direct API call: {e}")
            # Return user-friendly error message instead of actual error
            return "无法分析图像。请确保您上传了清晰的图像，并检查网络连接。 (Unable to analyze the image. Please ensure you've uploaded a clear image and check your network connection.)"
    
    def _convert_messages_to_openai_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        message_dicts = []
        for message in messages:
            if isinstance(message, HumanMessage) and isinstance(message.content, list):
                # Handle multimodal messages (with images)
                # Ensure each content item is properly formatted
                content_list = []
                for item in message.content:
                    if isinstance(item, dict):
                        if "type" in item and "text" in item and item["type"] == "text":
                            content_list.append({"type": "text", "text": item["text"]})
                            logger.info(f"Added text content: {item['text'][:50]}...")
                        elif "type" in item and "image_url" in item and item["type"] == "image_url":
                            # Ensure image_url is correctly formatted
                            img_url = item["image_url"]
                            if isinstance(img_url, dict) and "url" in img_url:
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": img_url
                                })
                                # Log image info but not the full base64 data
                                url_start = img_url["url"][:30]
                                logger.info(f"Added image with URL starting: {url_start}...")
                                
                                # Log if it contains base64 data
                                if "base64" in img_url["url"]:
                                    logger.info("Image contains base64 data")
                                    # Get the MIME type
                                    mime_match = re.search(r"data:(.*?);base64,", img_url["url"])
                                    if mime_match:
                                        logger.info(f"Image MIME type: {mime_match.group(1)}")
                            else:
                                # Handle direct URL string
                                url_to_use = {"url": img_url}
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": url_to_use
                                })
                                logger.info(f"Added image with direct URL: {img_url[:30]}...")
                message_dict = {"role": "user", "content": content_list}
                logger.info(f"Created multimodal message with {len(content_list)} content items")
            else:
                # Handle text-only messages
                message_dict = {"role": "", "content": message.content}
                if isinstance(message, HumanMessage):
                    message_dict["role"] = "user"
                elif isinstance(message, AIMessage):
                    message_dict["role"] = "assistant"
                elif isinstance(message, SystemMessage):
                    message_dict["role"] = "system"
                else:
                    raise ValueError(f"Unsupported message type: {type(message)}")
            message_dicts.append(message_dict)
        
        return message_dicts
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> CustomChatGeneration:
        """Generate a response using direct OpenAI client."""
        logger.info(f"Generating with model {self.model_name}")
        try:
            openai_messages = self._convert_messages_to_openai_format(messages)
            logger.info(f"Converted {len(openai_messages)} messages to OpenAI format")
            
            content = self._make_direct_api_call(
                messages=openai_messages,
                model=self.model_name,
                temperature=self.temperature,
                stop=stop
            )
            
            # If the response contains our error message, log it but still return a valid response
            if content == "无法分析图像。请确保您上传了清晰的图像，并检查网络连接。 (Unable to analyze the image. Please ensure you've uploaded a clear image and check your network connection.)":
                logger.warning("Returning friendly error message from API call")
            
            return CustomChatGeneration(
                message=AIMessage(content=content),
                generation_info={"finish_reason": "stop"},
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"API Key provided: {'Yes (length: ' + str(len(self.api_key)) + ')' if self.api_key else 'No'}")
            logger.error(f"Base URL provided: {self.base_url or 'No (using default)'}")
            logger.error(f"Model requested: {self.model_name}")
            # Return a user-friendly error message
            return CustomChatGeneration(
                message=AIMessage(content="无法分析图像。请确保您上传了清晰的图像，并检查网络连接。 (Unable to analyze the image. Please ensure you've uploaded a clear image and check your network connection.)"),
                generation_info={"finish_reason": "error"},
            ) 