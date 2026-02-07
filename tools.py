# tools.py

import os
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
