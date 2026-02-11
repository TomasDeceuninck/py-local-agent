import os
import json
import sys
from datetime import datetime
from collections import defaultdict

# Add the parent directory (project root) to sys.path to allow importing agent and config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import Agent
from config import OLLAMA_MODEL, OLLAMA_VISION_MODEL
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

# Load ground truth descriptions
GROUND_TRUTH_DESCRIPTIONS = {}
for i in range(1, 3): # For image-1 and image-2
    try:
        with open(f"evaluations/images/image-{i}-description.txt", 'r', encoding='utf-8') as f:
            GROUND_TRUTH_DESCRIPTIONS[f"description for image {i}"] = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Ground truth description for image-{i} not found.")

def compare_tool_calls(expected, actual):
    """
    Compares expected tool calls with actual tool calls.
    Returns True if all expected tool calls are found in actual, False otherwise.
    This function is now more robust to check for exact matches.
    """
    if not expected and not actual:
        return True # No tool calls expected, none made

    if len(expected) != len(actual):
        return False # Number of tool calls mismatch

    for exp_call in expected:
        found_match = False
        for act_call in actual:
            # Check name
            if exp_call.get("name") == act_call.get("name"):
                # Check arguments - full match required
                if exp_call.get("args") == act_call.get("args"):
                    found_match = True
                    break
        if not found_match:
            return False
    return True

def evaluate_response_with_llm(llm_evaluator, agent_response, expected_content):
    """
    Uses an LLM to evaluate if the agent's response is correct/relevant based on expected content.
    Returns True/False and the LLM's reasoning.
    """
    # Replace placeholder with actual ground truth description if applicable
    if expected_content in GROUND_TRUTH_DESCRIPTIONS:
        expected_content = GROUND_TRUTH_DESCRIPTIONS[expected_content]

    prompt = f"""You are an impartial AI judge. Your task is to evaluate an agent's response based on a given expected content.

Agent's Response: "{agent_response}"
Expected Content/Ground Truth: "{expected_content}"

Is the Agent's Response correct and relevant compared to the Expected Content?
Focus on the factual accuracy and relevance. It doesn't have to be an exact word-for-word match, but the core information should be present and accurate.

Respond with "[[YES]]" if it is correct and relevant, or "[[NO]]" if it is not.
After your YES/NO, provide a brief explanation of your reasoning.
"""
    try:
        llm_response = llm_evaluator.invoke([HumanMessage(content=prompt)])
        llm_response_content = llm_response.content.strip()
        
        if llm_response_content.startswith("[[YES]]"):
            return True, llm_response_content
        else:
            return False, llm_response_content
    except Exception as e:
        return False, f"Error during LLM evaluation: {e}"

def run_evaluation():
    print(f"{COLOR_BLUE}--- Starting Agent Evaluation ---{COLOR_RESET}")
    start_time = datetime.now()

    agent = Agent() # Initialize the agent once for all tests
    # Initialize a separate LLM for evaluation
    llm_evaluator = ChatOllama(model=OLLAMA_MODEL, temperature=0) # Using agent's main LLM for evaluation

    scenarios_dir = "evaluations/scenarios"
    scenario_files = [f for f in os.listdir(scenarios_dir) if f.endswith(".json")]

    total_tests = 0
    passed_tests = 0
    results = {}
    
    # Store initial SystemMessage to reset agent for each test
    initial_system_message_content = agent.chat_history[0].content

    for scenario_file in sorted(scenario_files):
        scenario_path = os.path.join(scenarios_dir, scenario_file)
        print(f"\\n{COLOR_YELLOW}Loading scenario: {scenario_file}{COLOR_RESET}")
        
        with open(scenario_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        scenario_results = []
        for i, test_case in enumerate(test_cases):
            total_tests += 1
            test_name = test_case.get("name", f"Unnamed Test Case {i+1}")
            user_input = test_case.get("user_input")
            expected_response_placeholder = test_case.get("expected_response_contains", "") # This might be a placeholder
            expected_tool_calls = test_case.get("expected_tool_calls", [])

            print(f"  {COLOR_BLUE}Running test: {test_name}{COLOR_RESET}")
            print(f"    User Input: {user_input}")

            try:
                # Reset chat history for each test case for isolation
                agent.chat_history = [
                    SystemMessage(content=initial_system_message_content)
                ]
                
                final_response_content, actual_tool_calls = agent.process_message(user_input)
                
                # Model-based evaluation for response content
                response_is_correct, llm_reasoning = evaluate_response_with_llm(
                    llm_evaluator, final_response_content, expected_response_placeholder
                )
                tool_calls_match = compare_tool_calls(expected_tool_calls, actual_tool_calls)

                is_passed = response_is_correct and tool_calls_match
                
                if is_passed:
                    passed_tests += 1
                    status = f"{COLOR_GREEN}PASSED{COLOR_RESET}"
                else:
                    status = f"{COLOR_RED}FAILED{COLOR_RESET}"
                
                print(f"    Agent Response: {final_response_content}")
                print(f"    Actual Tool Calls: {actual_tool_calls}")
                print(f"    Expected Tool Calls: {expected_tool_calls}")
                print(f"    Response Correct (LLM): {response_is_correct} - Reason: {llm_reasoning}")
                print(f"    Tool Calls Match: {tool_calls_match}")
                print(f"    Status: {status}\\n")

                scenario_results.append({
                    "name": test_name,
                    "user_input": user_input,
                    "final_response": final_response_content,
                    "actual_tool_calls": actual_tool_calls,
                    "expected_tool_calls": expected_tool_calls,
                    "expected_response_evaluated_against": (
                        GROUND_TRUTH_DESCRIPTIONS.get(expected_response_placeholder, expected_response_placeholder)
                    ),
                    "response_is_correct_llm": response_is_correct,
                    "llm_reasoning": llm_reasoning,
                    "tool_calls_match": tool_calls_match,
                    "status": "PASSED" if is_passed else "FAILED"
                })

            except Exception as e:
                status = f"{COLOR_RED}ERROR{COLOR_RESET}"
                print(f"    Error during test execution: {e}")
                print(f"    Status: {status}\\n")
                scenario_results.append({
                    "name": test_name,
                    "user_input": user_input,
                    "error": str(e),
                    "status": "ERROR"
                })
        results[scenario_file] = scenario_results

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"{COLOR_BLUE}--- Evaluation Summary ---{COLOR_RESET}")
    print(f"Total Test Files: {len(scenario_files)}")
    print(f"Total Test Cases: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")
    print(f"Pass Rate: {passed_tests / total_tests * 100:.2f}%")
    print(f"Duration: {duration}")

    # Optionally, save detailed results to a file
    # with open(f"evaluations/results_{end_time.strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()