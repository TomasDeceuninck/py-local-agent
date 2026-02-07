# main.py
# 1. Make sure your virtual environment is active
#
# 2. Install dependencies:
#    pip install -r requirements.txt
#
# 3. Run the agent:
#    python main.py

from langchain_ollama.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- Setup ---
# Use the gemma3:4b model as requested.
llm = ChatOllama(model="mistral")

# Initialize the search tool
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Bind the tools to the LLM. This allows the LLM to "see" the tools.
llm_with_tools = llm.bind_tools(tools)

# A list to store the conversation history
chat_history = []

print("Ollama Tool-Using Agent is ready. Type 'exit' to end the conversation.")

# --- Main Loop ---
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append the user's message to the history
    chat_history.append(HumanMessage(content=user_input))

    # First call to the model
    # The model can respond with a message or a tool call
    ai_response = llm_with_tools.invoke(chat_history)

    # If the model did NOT call a tool, just print the answer
    if not ai_response.tool_calls:
        print(f"Agent: {ai_response.content}")
        chat_history.append(ai_response) # Add the response to history
        continue

    # If the model DID call a tool, we need to handle it
    print("Agent wants to use a tool...")
    chat_history.append(ai_response) # Add the tool-calling response to history

    # Execute the tool calls and gather the results
    tool_outputs = []
    for tool_call in ai_response.tool_calls:
        print(f"-> Calling tool '{tool_call['name']}' with args {tool_call['args']}")
        tool_output = search_tool.invoke(tool_call["args"])
        tool_outputs.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        )

    # Add the tool outputs to the conversation history
    chat_history.extend(tool_outputs)

    # Second call to the model
    # This time, we provide the tool's output so the model can generate a final answer
    print("...sending tool results back to Agent to get a final answer.")
    final_response = llm_with_tools.invoke(chat_history)

    print(f"Agent: {final_response.content}")
    chat_history.append(final_response) # Add the final response to history