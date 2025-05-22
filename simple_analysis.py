import base64
import os
import sys
import argparse
import requests
from dotenv import load_dotenv

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, language="en"):
    """Send image to GPT-4o for psychological analysis"""
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Encode image
    image_b64 = encode_image(image_path)
    
    # Select prompt based on language
    if language == "zh":
        prompt = """作为一名专业心理学家，请分析这张HTP(房子-树-人)测试图。
        
分析内容应该包括：
1. 图像中的视觉特征和它们的心理学意义
2. 情绪状态的指标
3. 认知功能的评估
4. 人格特质的表现
5. 潜在的心理需求或困扰
6. 积极方面和成长潜力

请提供专业、平衡的分析，避免过度解读或下定论。以清晰的格式组织你的回答。"""
    else:
        prompt = """As a professional psychologist, please analyze this House-Tree-Person (HTP) test drawing.
        
Your analysis should include:
1. Visual features in the image and their psychological significance
2. Indicators of emotional state
3. Assessment of cognitive functioning
4. Personality traits displayed
5. Potential psychological needs or concerns
6. Positive aspects and growth potential

Please provide a professional, balanced analysis while avoiding overinterpretation or definitive conclusions. Organize your response in a clear format."""

    # Prepare API request
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Prepare the message with image
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }
    
    # Make the API call
    try:
        print("Sending image to GPT-4o for analysis...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract and return the content
        result = response.json()
        analysis = result["choices"][0]["message"]["content"]
        return analysis
    
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        if hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        sys.exit(1)

def save_analysis(analysis, output_path):
    """Save analysis to a file"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"Analysis saved to {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple HTP Drawing Analyzer using GPT-4o")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--output", type=str, default="analysis_result.txt", help="Path to save the analysis")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"], help="Language for analysis (en/zh)")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found")
        sys.exit(1)
    
    # Analyze the image
    analysis = analyze_image(args.image, args.language)
    
    # Print the analysis
    print("\n=== Psychological Analysis ===\n")
    print(analysis)
    
    # Save the analysis
    save_analysis(analysis, args.output)

if __name__ == "__main__":
    main() 