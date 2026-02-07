# main.py

from agent import Agent

def main():
    """
    Main function to run the chat loop.
    """
    # Initialize the agent
    agent = Agent()
    
    # --- Main Loop ---
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Process the user's message using the agent
        response = agent.process_message(user_input)
        
        print(f"Agent: {response}")

if __name__ == "__main__":
    main()
