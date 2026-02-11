# main.py

from agent import Agent

# Define ANSI escape codes for colors
COLOR_CYAN = "\033[96m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"

def main():
    """
    Main function to run the chat loop.
    """
    # Initialize the agent
    agent = Agent()
    
    try:
        # --- Main Loop ---
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Process the user's message using the agent
            response = agent.process_message(user_input)
            
            # Print agent's response in CYAN color
            print(f"{COLOR_CYAN}Agent: {response}{COLOR_RESET}")
    finally:
        # Ensure color is reset even if program crashes or exits unexpectedly
        print(COLOR_RESET, end="")

if __name__ == "__main__":
    main()