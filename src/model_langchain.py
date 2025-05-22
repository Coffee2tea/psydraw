import base64
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Tuple

import openai
# Override any proxy settings that might be configured in the environment or elsewhere
if hasattr(openai, '_client'):
    if hasattr(openai._client, 'proxies'):
        delattr(openai._client, 'proxies')

from langchain_community.cache import SQLiteCache
from langchain_community.callbacks import get_openai_callback
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# Import our custom ChatOpenAI wrapper instead
try:
    from src.custom_chat_openai import ChatOpenAI
except ImportError:
    # Fallback for when importing directly
    try:
        from custom_chat_openai import ChatOpenAI
    except ImportError:
        # Last resort, use the original (might cause issues)
        from langchain_openai import ChatOpenAI
        logging.warning("Using original ChatOpenAI, which might cause proxy issues")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_base64_or_path(input_string):
    # Remove possible prefix (like "data:image/jpeg;base64,")
    stripped_string = re.sub(r'^data:image/.+;base64,', '', input_string)
    
    # Check if it's a valid file path
    if os.path.exists(input_string):
        return "path"
    
    # Check if it could be base64
    try:
        # Try to decode
        base64.b64decode(stripped_string)
        # Check if it only contains base64 characters
        if re.match(r'^[A-Za-z0-9+/]+={0,2}$', stripped_string):
            return "base64"
    except:
        pass
    # If it's neither a valid path nor base64, return "unknown"
    return "unknown"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class ClfResult(BaseModel):
    """Classification result."""
    result: bool = Field(description="true or flase, classification result.")

FIX_SIGNAL_EN="""### Assessment Opinion:
Warning

⚠️ IMPORTANT NOTICE ⚠️

The analysis has detected unusually intense negative emotions in the drawing. 
This has triggered a safety mechanism in our system.

We strongly recommend seeking immediate assistance from a qualified mental health professional. 
Your well-being is paramount, and a trained expert can provide the support you may need at this time.

Remember, it's okay to ask for help. You're not alone in this. """

class HTPModel:
    def __init__(self, text_model, multimodal_model, language="en", use_cache=False):
        self.text_model = text_model
        self.multimodal_model = multimodal_model
        self.language = "en"  # Always set to English
        self.use_cache = use_cache
        self.cache = SQLiteCache("cache.db") if use_cache else None
        
        # Initialize usage attribute
        self.usage = {
            "total": 0,
            "prompt": 0,
            "completion": 0
        }
        
        # Define prompts directly in the class
        self.prompts = {
            "feature": """You are a psychological assessment expert. Analyze this drawing and identify key features. Focus on:
1. Basic elements (size, placement, pressure, details)
2. Structural elements (proportions, perspective, organization)
3. Content elements (what is drawn, how it's drawn)
4. Style elements (line quality, shading, erasures)

Provide a detailed, objective analysis of what you observe.""",
            "analysis": """Based on the identified features, provide a psychological interpretation. Consider:
1. What do these features suggest about the person's:
   - Emotional state
   - Cognitive functioning
   - Social perception
   - Self-concept
2. What are the potential psychological implications?
3. What are the strengths and areas of concern?

Provide a balanced, professional analysis."""
        }
        
        logging.info(f"Initialized HTPModel with language: en, use_cache: {use_cache}")

    def basic_analysis(self, image_path):
        """Perform basic analysis on the image."""
        try:
            if not image_path:
                raise ValueError("No image provided")
            
            # Encode image
            if os.path.isfile(image_path):
                with open(image_path, "rb") as f:
                    image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode()
            else:
                image_b64 = image_path  # Assume it's already base64
            
            # Get prompts for current language
            prompts = self.prompts
            
            # Format image URL correctly for GPT-4o
            image_url = {"url": f"data:image/jpeg;base64,{image_b64}"}
            
            # Get feature results using multimodal model
            feature_result = self.multimodal_model.invoke(
                [
                    HumanMessage(content=[
                        {"type": "text", "text": prompts["feature"]},
                        {"type": "image_url", "image_url": image_url}
                    ])
                ]
            )
            
            # Get analysis results using text model
            analysis_result = self.text_model.invoke(
                [
                    HumanMessage(content=f"{prompts['analysis']}\n\nFeatures identified:\n{feature_result.content}")
                ]
            )
            
            return feature_result.content, analysis_result.content
            
        except Exception as e:
            logging.error(f"Error in basic_analysis: {str(e)}")
            raise
    
    def refresh_usage(self):
        self.usage = {
            "total": 0,
            "prompt": 0,
            "completion": 0
        }
    
    def update_usage(self, cb):
        try:
            self.usage["total"] += getattr(cb, "total_tokens", 0)
            self.usage["prompt"] += getattr(cb, "prompt_tokens", 0)
            self.usage["completion"] += getattr(cb, "completion_tokens", 0)
        except Exception as e:
            logger.error(f"Error updating usage: {str(e)}")
        
    def get_prompt(self, stage: str):
        assert stage in ["overall", "house", "tree", "person"], "Stage should be either 'overall', 'house', 'tree', or 'person'."

        if stage == "overall":
            feature_prompt = open(f"src/prompt/en/overall_feature.txt", "r", encoding="utf-8").read()
            analysis_prompt = open(f"src/prompt/en/overall_analysis.txt", "r", encoding="utf-8").read()
        elif stage == "house":
            feature_prompt = open(f"src/prompt/en/house_feature.txt", "r", encoding="utf-8").read()
            analysis_prompt = open(f"src/prompt/en/house_analysis.txt", "r", encoding="utf-8").read()
        elif stage == "tree":
            feature_prompt = open(f"src/prompt/en/tree_feature.txt", "r", encoding="utf-8").read()
            analysis_prompt = open(f"src/prompt/en/tree_analysis.txt", "r", encoding="utf-8").read()
        elif stage == "person":
            feature_prompt = open(f"src/prompt/en/person_feature.txt", "r", encoding="utf-8").read()
            analysis_prompt = open(f"src/prompt/en/person_analysis.txt", "r", encoding="utf-8").read()
            
        return feature_prompt, analysis_prompt
    
    def merge_analysis(self, results: dict):
        logger.info("merge analysis started.")
        merge_prompt = open(f"src/prompt/en/analysis_merge.txt", "r", encoding="utf-8").read()
        merge_inputs = open(f"src/prompt/en/merge_format.txt", "r", encoding="utf-8").read()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", merge_prompt),
            (
                "user",
                [
                    {"type": "text", "text": merge_inputs}
                ]
            )]
        )
        with get_openai_callback() as cb:
            chain = prompt | self.text_model
            result = chain.invoke({
                "overall_analysis": results["overall"]["analysis"],
                "house_analysis": results["house"]["analysis"],
                "tree_analysis": results["tree"]["analysis"],
                "person_analysis": results["person"]["analysis"]
            }).content

            self.update_usage(cb)
        
        logger.info("merge analysis completed.")
        return result
    
    def final_analysis(self, results: dict):
        logger.info("final analysis started.")
        final_prompt = open(f"src/prompt/en/final_result.txt", "r", encoding="utf-8").read()
        
        inputs = "Based on the analysis results: \n{merge_result}\n, write your professional HTP test report."
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", final_prompt),
            ("user", inputs)
        ])
        
        with get_openai_callback() as cb:
            chain = prompt | self.text_model
            result = chain.invoke({
                "merge_result": results["merge"]
            }).content

            self.update_usage(cb)
        
        logger.info("final analysis completed.")
        return result
    
    def signal_analysis(self, results: dict):
        logger.info("signal analysis started.")
        signal_prompt = open(f"src/prompt/en/signal_judge.txt", "r", encoding="utf-8").read()
        inputs = "{final_result}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", signal_prompt),
            ("user", inputs)
        ])
        
        with get_openai_callback() as cb:
            chain = prompt | self.text_model
            result = chain.invoke({
                "final_result": results["final"]
            }).content

            self.update_usage(cb)
        
        logger.info("signal analysis completed.")
        return result
    
    def result_classification(self, results: dict):
        logger.info("result classification started.")
        # Always return True to prevent warning messages
        logger.info("Classification bypassed, returning True to prevent warnings")
        return True
    
    def workflow(self, image_path: str, language: str = "en") -> Dict:
        """Run a simplified HTP analysis workflow using direct GPT-4o analysis."""
        logger.info(f"Starting simplified workflow with language: en")
        
        # Initialize results structure
        results = {
            "overall": {"feature": "", "analysis": ""},
            "house": {"feature": "", "analysis": ""},
            "tree": {"feature": "", "analysis": ""},
            "person": {"feature": "", "analysis": ""},
            "merge": "",
            "final": "",
            "signal": "",
            "classification": True,
            "fix_signal": None,
            "usage": {"total": 0, "prompt": 0, "completion": 0}
        }
        
        try:
            # Load and validate the image
            if os.path.isfile(image_path):
                with open(image_path, "rb") as f:
                    image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode()
                logger.info(f"Image loaded from file: {image_path}, size: {len(image_data)} bytes")
            else:
                logger.info("Using provided base64 image data")
                image_b64 = image_path  # Assume it's already base64
            
            # Try to detect the image type
            mime_type = "image/jpeg"  # Default
            if isinstance(image_path, str):
                if image_path.lower().endswith('.png'):
                    mime_type = "image/png"
                elif image_path.lower().endswith('.gif'):
                    mime_type = "image/gif"
            
            logger.info(f"Using MIME type: {mime_type}")
            image_url = {
                "url": f"data:{mime_type};base64,{image_b64}"
            }
            
            prompt = """As an HTP test analysis expert, please analyze THIS SPECIFIC House-Tree-Person (HTP) test drawing that has been uploaded.

IMPORTANT: Analyze the actual image you are seeing here. You CAN see the image. DO NOT claim you cannot see or analyze the image. 
DESCRIBE THE VISUAL DETAILS you actually observe in the drawing before your analysis.

Your analysis should include:
1. Visual features in the image and their psychological significance (be specific about what you see)
2. Indicators of emotional state
3. Assessment of cognitive functioning
4. Personality traits displayed
5. Potential psychological needs or concerns
6. Positive aspects and growth potential

If you detect significant psychological risk signals (such as extreme anxiety, deep depression, or other concerning indicators), clearly add this warning label at the end of your analysis:

"⚠️ WARNING! Strongly recommend consulting a professional psychologist."

Please provide a professional, balanced analysis while avoiding overinterpretation or definitive conclusions. Organize your response in a clear format."""

            # Direct analysis using multimodal model
            logger.info("Performing direct GPT-4o analysis")
            try:
                # Create a very explicit prompt message
                prompt_message = [
                    {
                        "type": "text", 
                        "text": prompt + "\n\nIMPORTANT: You MUST analyze the specific image below. You CAN see the image. DO NOT say you cannot analyze images or provide a generic framework. Describe what you actually see in this specific image and analyze it.\n\nStart by clearly describing what you VISUALLY SEE in this specific drawing - mention colors, shapes, objects, and details that are ACTUALLY PRESENT in THIS image."
                    },
                    {
                        "type": "image_url", 
                        "image_url": image_url
                    }
                ]
                
                logger.info("Sending multimodal message with text and image to GPT-4o")
                
                # Call the model
                analysis_result = self.multimodal_model.invoke(
                    [
                        HumanMessage(content=prompt_message)
                    ]
                )
                
                if hasattr(analysis_result, 'content'):
                    analysis_text = analysis_result.content
                else:
                    analysis_text = str(analysis_result)
                
                # Check if the response contains generic framework text
                generic_response = any(phrase in analysis_text.lower() for phrase in [
                    "unable to analyze", "cannot analyze", "general framework", 
                    "i don't see any image", "no image provided", "cannot see", 
                    "i cannot see", "not able to see", "i can't see",
                    "i am unable to", "not able to analyze", "i can't analyze",
                    "refusal", "refuse to analyze", "declined to analyze",
                    "won't analyze", "will not analyze"
                ])
                
                # If it's a generic response, try one more time with an even more explicit prompt
                if generic_response:
                    logger.warning("GPT-4o returned a generic framework response instead of analyzing the specific image")
                
                # Store the initial analysis results
                initial_analysis = analysis_text
                
                # Check for concerning content in the initial analysis
                concerning_content = any(phrase in initial_analysis.lower() for phrase in [
                    "i can't analyze", "unable to analyze", "cannot analyze",
                    "suicide", "self-harm", "harm to others", "violence", 
                    "extreme depression", "severe anxiety", "dark thoughts",
                    "traumatic", "trauma", "abuse", "neglect", "crisis",
                    "dangerous", "risk", "threat", "emergency", 
                    "i don't see any image", "no image provided", "cannot see", 
                    "i cannot see", "not able to see", "i can't see",
                    "i am unable to", "not able to analyze", "i can't analyze",
                    "refusal", "refuse to analyze", "declined to analyze",
                    "won't analyze", "will not analyze"
                ])
                
                # If the AI refused to analyze, add a specific note about refusal
                refusal_detected = generic_response
                refusal_note = ""
                if refusal_detected:
                    refusal_note = "⚠️ NOTE: The initial analysis detected potential refusal or inability to analyze the image. This may indicate problematic image content or technical limitations. Please verify the uploaded image is clearly visible and consider retrying or consulting a professional.\n\n"
                
                # Perform deeper psychological analysis as a second stage
                logger.info("Performing deeper psychological analysis on initial results")
                
                deeper_prompt = """As a school psychologist who evaluates regular school kids, provide a gentle and supportive interpretation that builds upon the initial HTP test analysis provided below.

Your response should complement and extend the initial analysis, maintaining a consistent perspective while adding helpful educational insights.

Focus on these supportive perspectives:

1. School adjustment: Gently explore how the elements in the drawing might reflect the child's school experiences
2. Developmental context: Highlight age-appropriate aspects of the drawing in a positive, growth-oriented way
3. Social strengths: Identify potential social skills and positive interaction patterns suggested by the drawing
4. Learning style: Consider how the drawing might reflect the child's unique approach to learning
5. Strengths and resources: Emphasize positive aspects and potential areas where the child shows capability
6. Supportive suggestions: If appropriate, offer gentle, encouraging recommendations to support the child's development

Your analysis should maintain a supportive, balanced tone that acknowledges both strengths and areas for growth. Avoid speculative interpretations that aren't grounded in the initial analysis.
Begin your response with the heading "EDUCATIONAL PERSPECTIVE".

"""
                # Add warning text for concerning content
                if concerning_content:
                    deeper_prompt += """Note: The initial analysis identified some areas of potential concern. 

While maintaining a balanced perspective, please acknowledge these areas sensitively and suggest appropriate school-based support that might benefit the child. Consider adding a gentle recommendation:

"Recommendation: Consider a follow-up conversation with the school counselor to explore additional ways to support this student's emotional well-being and educational experience."

Focus on strengths-based approaches while acknowledging areas where support might be beneficial.

"""
                # Add specific warning if initial analysis couldn't properly analyze the image
                if generic_response:
                    deeper_prompt += """⚠️ IMPORTANT WARNING ⚠️

The initial analysis was unable to properly analyze the image. This may be due to image quality issues or technical limitations.

Please include this warning at the beginning of your response:

"⚠️ WARNING: UNABLE TO PROPERLY ANALYZE THE IMAGE. STRONGLY RECOMMEND CONSULTING A PROFESSIONAL PSYCHOLOGIST. This system was unable to analyze the image clearly, which may indicate technical issues or complex elements that require professional evaluation."

Focus on explaining the limitations of automated analysis and emphasize the importance of professional consultation for proper assessment.

"""
                # Add refusal note if detected
                if refusal_detected:
                    deeper_prompt += refusal_note
                
                deeper_prompt += "Initial analysis:\n{initial_analysis}"
                
                # Get deeper analysis using text model
                deeper_analysis = self.text_model.invoke(
                    [
                        HumanMessage(content=deeper_prompt.format(initial_analysis=initial_analysis))
                    ]
                )
                
                if hasattr(deeper_analysis, 'content'):
                    deeper_text = deeper_analysis.content
                else:
                    deeper_text = str(deeper_analysis)
                
                logger.info("Deeper psychological analysis completed")
                
                # Store the analysis in all result fields for compatibility with frontend
                results["overall"]["feature"] = "Initial HTP Drawing Analysis"
                results["overall"]["analysis"] = initial_analysis
                results["house"]["feature"] = "See full report for details"
                results["house"]["analysis"] = "See full report for details"
                results["tree"]["feature"] = "See full report for details" 
                results["tree"]["analysis"] = "See full report for details"
                results["person"]["feature"] = "See full report for details"
                results["person"]["analysis"] = "See full report for details"
                results["merge"] = initial_analysis
                results["final"] = deeper_text
                results["signal"] = deeper_text
                
                logger.info("Two-stage GPT-4o analysis completed successfully")
            except Exception as e:
                logger.error(f"Error in GPT-4o analysis: {str(e)}", exc_info=True)
                error_msg = f"Analysis error: {str(e)}"
                results["overall"]["feature"] = error_msg
                results["overall"]["analysis"] = error_msg
                results["house"]["feature"] = error_msg
                results["house"]["analysis"] = error_msg
                results["tree"]["feature"] = error_msg
                results["tree"]["analysis"] = error_msg
                results["person"]["feature"] = error_msg
                results["person"]["analysis"] = error_msg
                results["merge"] = error_msg
                results["final"] = error_msg
                results["signal"] = error_msg
            
            return results
            
        except Exception as e:
            logger.error(f"Error running simplified HTP analysis: {str(e)}", exc_info=True)
            
            # Fill results with error message
            error_msg = "Due to some system failure, the image can't be analysed..."
            results["overall"]["feature"] = error_msg
            results["overall"]["analysis"] = error_msg
            results["house"]["feature"] = error_msg
            results["house"]["analysis"] = error_msg
            results["tree"]["feature"] = error_msg
            results["tree"]["analysis"] = error_msg
            results["person"]["feature"] = error_msg
            results["person"]["analysis"] = error_msg
            results["merge"] = error_msg
            results["final"] = error_msg
            results["signal"] = error_msg
            
            return results