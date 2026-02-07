# tools.py

import os
import base64
from PIL import Image # type: ignore
from io import BytesIO
from langchain_core.tools import tool
import numexpr

# Define the project root for security checks
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

@tool
def describe_image(image_path: str, prompt: str = "Describe this image.") -> str:
    """
    Analyzes an image file and provides a description or answers a question about it.
    Useful for understanding the content of local images.
    Input must be a valid image file path relative to the project root.
    The 'prompt' argument can be used to ask a specific question about the image.
    """
    abs_image_path = os.path.abspath(os.path.join(PROJECT_ROOT, image_path))

    # Security check: Ensure the file is within the project root
    if not os.path.commonpath([PROJECT_ROOT, abs_image_path]) == PROJECT_ROOT:
        return f"Error: Access denied. Cannot process image outside project directory: {image_path}"

    try:
        # Load image and convert to base64
        with Image.open(abs_image_path) as img:
            buffered = BytesIO()
            # Save as JPEG for consistent base64 encoding for multimodal models
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # The actual vision model invocation will happen in agent.py
        # This tool just prepares the data and signals its use
        return f"Image '{image_path}' ready for vision model with prompt: '{prompt}'. " \
               f"Please refer to the vision model's output for analysis."
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error processing image {image_path}: {e}"