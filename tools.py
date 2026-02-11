# tools.py

import os
import base64
import pyttsx3 # Added for TTS
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import numexpr

# Define the project root for security checks
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

@tool
def read_file(file_path: str) -> str:
    """
    Reads the content of a file from the project directory.
    Useful for getting information from local files like documentation, code, or data.
    Input must be a valid file path relative to the project root.
    """
    abs_file_path = os.path.abspath(os.path.join(PROJECT_ROOT, file_path))

    # Security check: Ensure the file is within the project root
    if not os.path.commonpath([PROJECT_ROOT, abs_file_path]) == PROJECT_ROOT:
        return f"Error: Access denied. Cannot read file outside project directory: {file_path}"

    try:
        with open(abs_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Useful for performing calculations.
    Input must be a valid mathematical expression string (e.g., "2 + 2 * 5").
    """
    try:
        # Use numexpr for safe and fast evaluation
        result = str(numexpr.evaluate(expression))
        return result
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

def _encode_image_to_base64(image_path):
    """Convert a local image file to a base64 string."""
    abs_image_path = os.path.abspath(os.path.join(PROJECT_ROOT, image_path))
    
    # Security check: Ensure the file is within the project root
    if not os.path.commonpath([PROJECT_ROOT, abs_image_path]) == PROJECT_ROOT:
        raise PermissionError(f"Access denied. Cannot read file outside project directory: {image_path}")

    if not os.path.exists(abs_image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    with open(abs_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@tool
def analyze_image(image_path: str, prompt: str = "What is in this image? Provide a detailed description.", model_name: str = "llama3.2-vision") -> str:
    """
    Analyzes a local image file using a vision model and returns a detailed description.
    Useful for understanding the content of images.
    Input must be a valid image file path relative to the project root.
    """
    try:
        # Initialize the local Ollama model
        llm = ChatOllama(model=model_name, temperature=0)

        # Encode your image
        base64_image = _encode_image_to_base64(image_path)

        # Create the multimodal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # Invoke the model
        response = llm.invoke([message])
        
        return response.content
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error analyzing image {image_path}: {e}"

@tool
def speak(text: str) -> str:
    """
    Converts the given text to speech and plays it aloud.
    Useful for having the agent communicate verbally.
    """
    try:
        engine = pyttsx3.init()
        # You can set properties like rate, volume, and voice here if needed
        # For example:
        # engine.setProperty('rate', 150)
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[0].id) # Use the first available voice
        engine.say(text)
        engine.runAndWait()
        return f"Successfully spoke: '{text}'"
    except Exception as e:
        return f"Error speaking text: {e}"