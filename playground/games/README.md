# Adversarial AI Chat Game

## Overview

This project implements an adversarial game between two AI chat models using Ollama. One model plays the role of an argumentative chatbot, while the other tries to be polite and agreeable. The conversation is logged in a file, and the interaction continues for a predefined number of exchanges.

## Features

- **Argumentative AI vs. Polite AI:** One model disagrees with everything, while the other tries to calm the conversation.
- **Automated Interaction:** The two models talk to each other without human intervention.
- **Conversation Logging:** All exchanges are stored in a `conversation.txt` file.
- **Customizable Models:** You can change the AI models and system prompts.

## Prerequisites

- Python 3.x
- [Ollama](https://ollama.ai) installed for local model execution

## Installation

1. Install Ollama:
   ```sh
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
2. Pull the required models (modify as needed):
   ```sh
   ollama pull llama3:8b
   ollama pull llama3.2:1b
   ```
3. Install Python dependencies:
   ```sh
   pip install ollama
   ```

## Usage

1. Run the script:
   ```sh
   python adversarial_chat.py
   ```
2. The conversation will be logged in `conversation.txt`.

## Customization

- Modify `MODEL_ONE` and `MODEL_TWO` to use different AI models.
- Change `model_one_system_prompt` and `model_two_system_prompt` to adjust personalities.
- Adjust `MAX_HISTORY` to control conversation length.

## Example Conversation

```
[2025-03-12 10:00:00] ASSISTANT: I think this is a great idea!
[2025-03-12 10:00:02] USER: No, that's the worst idea I've ever heard!
[2025-03-12 10:00:04] ASSISTANT: Well, I respect your opinion, but I still think it has some value.
[2025-03-12 10:00:06] USER: Value? Are you kidding me? This is absurd!
```
