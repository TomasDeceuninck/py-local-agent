# agent.py

import base64
from PIL import Image # type: ignore
from io import BytesIO
import os # Added for PROJECT_ROOT

from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from config import OLLAMA_MODEL, OLLAMA_VISION_MODEL
from tools import read_file, calculator, describe_image # Added describe_image

# Define ANSI escape codes for colors (DIM and RESET only for internal prints)
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"

# Define the project root for security checks (duplicated from tools.py for agent's internal use)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Agent:
    def __init__(self):
        """Initializes the agent with an LLM, tools, and chat history."""
        self.llm = ChatOllama(model=OLLAMA_MODEL)
        self.vision_llm = ChatOllama(model=OLLAMA_VISION_MODEL) # New vision LLM

        
        # Define the tools the agent can use
        self.search_tool = DuckDuckGoSearchRun()
        self.read_file_tool = read_file
        self.calculator_tool = calculator
        self.describe_image_tool = describe_image # New describe_image tool
        self.tools = [self.search_tool, self.read_file_tool, self.calculator_tool, self.describe_image_tool] # Added to tools

        # Bind the tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # A list to store the conversation history, starting with a system prompt
        self.chat_history = [
            SystemMessage(
                content="You are a helpful and resourceful AI assistant. "
                        "You have access to a set of powerful tools. "
                        "Your primary goal is to use these tools whenever necessary to fulfill the user's requests. "
                        "When you use a tool, process its output and provide a concise, natural language summary or answer to the user. "
                        "Do not respond with tool calls directly. If a tool is relevant, you MUST use it."
            )
        ]
        print("Ollama Agent is ready.")

    def process_message(self, user_input: str) -> str:
        """
        Processes a user's message, runs the agent loop, and returns the final response.
        """
        self.chat_history.append(HumanMessage(content=user_input))

        # First call to the model
        ai_response = self.llm_with_tools.invoke(self.chat_history)
        self.chat_history.append(ai_response)

        # If the model called a tool, handle it
        if ai_response.tool_calls:
            print(f"{COLOR_DIM}Agent wants to use a tool...{COLOR_RESET}")
            tool_outputs = []
            for tool_call in ai_response.tool_calls:
                print(f"{COLOR_DIM}-> Calling tool '{tool_call['name']}' with args {tool_call['args']}{COLOR_RESET}")
                
                if tool_call['name'] == 'duckduckgo_search':
                    tool_output = self.search_tool.invoke(tool_call["args"])
                    tool_outputs.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
                elif tool_call['name'] == 'read_file':
                    tool_output = self.read_file_tool.invoke(tool_call["args"])
                    tool_outputs.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
                elif tool_call['name'] == 'calculator':
                    tool_output = self.calculator_tool.invoke(tool_call["args"])
                    tool_outputs.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
                elif tool_call['name'] == 'describe_image':
                    image_path = tool_call['args'].get('image_path')
                    prompt = tool_call['args'].get('prompt', "Describe this image.")

                    if not image_path:
                        tool_outputs.append(
                            ToolMessage(content="Error: 'image_path' argument missing for describe_image tool.", tool_call_id=tool_call["id"])
                        )
                        continue
                    
                    abs_image_path = os.path.abspath(os.path.join(PROJECT_ROOT, image_path))
                    
                    # Security check: Ensure the file is within the project root
                    if not os.path.commonpath([PROJECT_ROOT, abs_image_path]) == PROJECT_ROOT:
                        tool_outputs.append(
                            ToolMessage(content=f"Error: Access denied. Cannot process image outside project directory: {image_path}", tool_call_id=tool_call["id"])
                        )
                        continue
                    
                    print(f"{COLOR_DIM}Resolved absolute image path: {abs_image_path}{COLOR_RESET}") # Debug print
                    try:
                        with Image.open(abs_image_path) as img:
                            buffered = BytesIO()
                            if img.mode == 'RGBA':
                                img = img.convert('RGB')
                            img.save(buffered, format="JPEG")
                            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                        # Construct multimodal message for the vision model
                        vision_message = HumanMessage(
                            content=[
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ]
                        )
                        
                        vision_response = self.vision_llm.invoke([vision_message])
                        tool_outputs.append(
                            ToolMessage(content=vision_response.content, tool_call_id=tool_call["id"])
                        )
                    except FileNotFoundError:
                        tool_outputs.append(
                            ToolMessage(content=f"Error: Image file not found at {image_path}", tool_call_id=tool_call["id"])
                        )
                    except Exception as e:
                        tool_outputs.append(
                            ToolMessage(content=f"Error processing image {image_path} with vision model: {e}", tool_call_id=tool_call["id"])
                        )
                else:
                    tool_outputs.append(
                        ToolMessage(content=f"Error: Unknown tool '{tool_call['name']}'", tool_call_id=tool_call["id"])
                    )

            
            self.chat_history.extend(tool_outputs)

            # Second call to the model with the tool's output
            print(f"{COLOR_DIM}...sending tool results back to Agent to get a final answer.{COLOR_RESET}")
            final_response = self.llm_with_tools.invoke(self.chat_history)
            self.chat_history.append(final_response)
            return final_response.content
        
        # If no tool was called, return the AI's direct response
        return ai_response.content
