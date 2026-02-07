# agent.py

from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config import OLLAMA_MODEL

class Agent:
    def __init__(self):
        """Initializes the agent with an LLM, tools, and chat history."""
        self.llm = ChatOllama(model=OLLAMA_MODEL)
        
        # Define the tools the agent can use
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [self.search_tool]

        # Bind the tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # A list to store the conversation history
        self.chat_history = []
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
            print("Agent wants to use a tool...")
            tool_outputs = []
            for tool_call in ai_response.tool_calls:
                print(f"-> Calling tool '{tool_call['name']}' with args {tool_call['args']}")
                
                # --- This is where you would add logic for more tools ---
                if tool_call['name'] == 'duckduckgo_search':
                    tool_output = self.search_tool.invoke(tool_call["args"])
                    tool_outputs.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
                # Add `elif tool_call['name'] == 'your_new_tool':` for new tools
            
            self.chat_history.extend(tool_outputs)

            # Second call to the model with the tool's output
            print("...sending tool results back to Agent to get a final answer.")
            final_response = self.llm_with_tools.invoke(self.chat_history)
            self.chat_history.append(final_response)
            return final_response.content
        
        # If no tool was called, return the AI's direct response
        return ai_response.content
