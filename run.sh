#!/bin/bash
# This script activates the virtual environment and starts the Python agent.

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting the agent..."
python main.py
