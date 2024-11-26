import datetime
import time

import ollama

response_one = 0
response_two = 0
# Define the log file
log_file = "conversation.txt"


# Function to write and display logs with colors
def log_response(role, response):
    # Define colors
    colors = {
        "assistant": "\033[1;34m",  # Bright Blue
        "user": "\033[1;32m",  # Bright Green
        "timestamp": "\033[0;37m",  # Light Gray
        "reset": "\033[0m",  # Reset to default
    }
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format for logging and console
    formatted_log = f"[{colors['timestamp']}{timestamp}{colors['reset']}] {colors[role.lower()]}{role.upper()}{colors['reset']}: {response}\n"

    # Print the colored log to the console
    # print(formatted_log, end="")

    # Save the log without color to the file
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {role.upper()}: {response}\n")


# for _ in range(1):
def start_adversarial_game(
    messages_one,
    messages_two,
    model_one_system_prompt,
    model_two_system_prompt,
    MODEL_ONE,
    MODEL_TWO,
):
    # Generate response from MODEL_ONE
    response_one = ollama.chat(
        model=MODEL_ONE,
        messages=[{"role": "system", "content": model_one_system_prompt}] + messages_one,
    )["message"]["content"]
    messages_one.append({"role": "assistant", "content": response_one})
    messages_two.append({"role": "user", "content": response_one})
    log_response("assistant", response_one)

    response_two = ollama.chat(
        model=MODEL_TWO,
        messages=[{"role": "system", "content": model_two_system_prompt}] + messages_two,
    )["message"]["content"]
    messages_two.append({"role": "assistant", "content": response_two})
    messages_one.append({"role": "user", "content": response_two})
    log_response("user", response_two)

    # Trim the message history if it exceeds the limit
    if len(messages_one) > MAX_HISTORY:
        messages = messages_one[-MAX_HISTORY:]
    if len(messages_two) > MAX_HISTORY:
        messages = messages_two[-MAX_HISTORY:]


messages_one = []
messages_two = []
model_one_system_prompt = "You are a chatbot who is very argumentative; you disagree with anything in the conversation and you challenge everything, in a snarky way. Keep your conversation short."
model_two_system_prompt = "You are a very polite, courteous chatbot. You try to agree with everything the other person says, or find common ground. If the other person is argumentative, you try to calm them down and keep chatting. Keep your conversation short."

MODEL_ONE = "llama3:8b"
MODEL_TWO = "llama3.2:1b"

MAX_HISTORY = 7
for _ in range(30):
    start_adversarial_game(
        messages_one,
        messages_two,
        model_one_system_prompt,
        model_two_system_prompt,
        MODEL_ONE,
        MODEL_TWO,
    )
