# Ollama Agent with Web Search

A simple Python AI agent that uses a local Ollama model (`mistral`) and a web search tool to answer questions.

## Getting Started

Follow these steps to set up and run the agent.

### 1. Set up Ollama

Ensure you have [Ollama](https://ollama.ai/) installed and running.
Then, pull the `mistral` model:
```bash
ollama pull mistral
```

### 2. Create and Activate Virtual Environment

It's recommended to use a Python virtual environment to manage dependencies.

```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
# source ./.venv/bin/activate  # On Linux/macOS
```

### 3. Install Dependencies

With your virtual environment activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Agent

The easiest way to run the agent is by using the provided startup scripts.

*   **On Windows:**
    ```bash
    run.bat
    ```

*   **On Linux/macOS:**
    First, make the script executable:
    ```bash
    chmod +x run.sh
    ```
    Then run it:
    ```bash
    ./run.sh
    ```

Alternatively, you can manually activate the virtual environment and run the Python script directly:
```bash
python main.py
```

Type your questions, and the agent will use the `mistral` model and a web search tool to respond. Type `exit` to quit.
