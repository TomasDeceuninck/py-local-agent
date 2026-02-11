# agent.py

import os 

from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from config import OLLAMA_MODEL, OLLAMA_VISION_MODEL
from tools import read_file, calculator, analyze_image, speak # Added speak tool

# Define ANSI escape codes for colors (DIM and RESET only for internal prints)
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"

# Define the project root for security checks (duplicated from tools.py for agent's internal use)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Agent:
    def __init__(self):
        """Initializes the agent with an LLM, tools, and chat history."""
        self.llm = ChatOllama(model=OLLAMA_MODEL)
        
        # Define the tools the agent can use
        self.search_tool = DuckDuckGoSearchRun()
        self.read_file_tool = read_file
        self.calculator_tool = calculator
        self.analyze_image_tool = analyze_image
        self.speak_tool = speak # Added speak tool
        self.tools = [self.search_tool, self.read_file_tool, self.calculator_tool, self.analyze_image_tool, self.speak_tool] # Added speak tool

        # Bind the tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # A list to store the conversation history, starting with a system prompt
        self.chat_history = [
            SystemMessage(
                content="You are a helpful and resourceful AI assistant. "
                        "You have access to a set of powerful tools. "
                        "Your primary goal is to use these tools whenever necessary to fulfill the user's requests. "
                        "When you are asked to analyze an image, you MUST use the 'analyze_image' tool. "
                        "The 'analyze_image' tool takes an 'image_path' and an optional 'prompt' and 'model_name'. "
                        "By default, the 'analyze_image' tool uses 'llama3.2-vision' as the model. "
                        "Always use the provided image path directly for analysis. "
                        "When you are asked to speak something out loud, you MUST use the 'speak' tool. "
                        "The 'speak' tool takes a 'text' argument, which is the exact text to be spoken. "
                        "Process the tool's output and provide a concise, natural language summary or answer to the user. "
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
                elif tool_call['name'] == 'analyze_image':
                    # Explicitly pass the OLLAMA_VISION_MODEL from config.py
                    tool_args = tool_call["args"]
                    tool_args["model_name"] = OLLAMA_VISION_MODEL
                    tool_output = self.analyze_image_tool.invoke(tool_args)
                    tool_outputs.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
                elif tool_call['name'] == 'speak':
                    tool_output = self.speak_tool.invoke(tool_call["args"])
                    tool_outputs.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
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