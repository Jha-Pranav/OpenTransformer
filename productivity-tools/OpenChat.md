```markdown
# Ollama - Your Local AI Companion 🎉

## Welcome to OpenChat

Ollama streamlines running large language models (LLMs) on your local machine, though terminal interactions can be somewhat rudimentary. To elevate this experience, a more intuitive interface is required that preserves conversation history and maintains context. Ollama's Python library serves as a solution; however, scripting it out to create a browser-based command with an alias for quick access enhances usability significantly. This method not only enriches the interaction but also permits effortless switching between different LLMs stored locally, ensuring each session’s context is retained across models.
```

## ![alt text](image.png)

## Getting Started: The Easy Part 💻

1. **Install Required Packages**:
   \* pip install gradio ollama requests
2. **Pick a Model**: Choose from models from the list of models downloaded in local! 🎉

## The Not-So-Fun Part: Future Work 💡

- **Multi-user Support**: Let's get this party started and invite more friends over!
- **Better Error Handling**: We're working on making errors a thing of the past. 😊

## The Aliases We Love 💻

Add the following line to your shell configuration file (e.g., .bashrc, .zshrc):

```
alias openchat="python /path/to/GradioUI.py"
```

Replace `/path/to/openchat.py` with the actual path to your `GradioUI.py` file. Now, you can execute OpenChat by simply typing `openchat` in your terminal - no more remembering command lines! 🙅‍♂️

### Code

```
#!/usr/bin/env python
# coding: utf-8

import gradio as gr
import ollama
import requests
import random
import webbrowser


MAX_HISTORY = 10
MODEL = "llama3:8b"

SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful AI assistant. Provide accurate, concise, and "
    "context-aware responses tailored to the user's needs. Maintain a professional, neutral tone, "
    "ask clarifying questions if needed, and avoid speculation or harmful content. "
    "Prioritize factual accuracy, explain concepts when necessary, and adapt to the user's "
    "familiarity with the topic. Return responses in the markdown format."
)
conversation = [{"role": "system", "content": SYSTEM_PROMPT}]


def llm(MODEL, input_text):
    # Initialize the conversation if it's the first message
    global conversation

    # Append user input to the conversation
    conversation.append({"role": "user", "content": input_text})

    # Get the response from the LLM
    stream_out = ollama.chat(model=MODEL, messages=conversation, stream=True)

    # Collect the assistant's response
    output = ""
    for stream in stream_out:
        output += stream["message"]["content"]
        yield output

    # Append the assistant's response to the conversation
    conversation.append({"role": "assistant", "content": output})
    # Trim the message history if it exceeds the limit
    if len(conversation) > MAX_HISTORY:
        messages = conversation[-MAX_HISTORY:]

model_list = requests.get("http://localhost:11434/api/tags").json()['models']
# Define the Gradio interface
interface = gr.Interface(
    fn=llm,
    inputs=[
        gr.Radio(
            choices=[(d.get('name'),d.get('model')) for d in model_list],
            value=random.choices([d.get('name') for d in model_list])[0],
            type="value",
        ),
        gr.Textbox(label="Your message:", lines=6),  # Input: User message
    ],
    outputs=[
        gr.Markdown(label="Response:",show_copy_button=True),  # Output: Assistant's response
    ],
    flagging_mode="never",
    theme=gr.Theme.from_hub("gstaff/sketch"),
    fill_width=True,
)


url = "http://127.0.0.1:7860"
print('Opening in the default browser')
webbrowser.open(url)

# Launch the Gradio interface
interface.launch(share=False ,server_port=7860)
```
