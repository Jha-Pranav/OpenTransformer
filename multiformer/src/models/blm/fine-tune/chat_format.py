def convert_stories_to_chat_format(text):
    text = "User:" + text
    text = text.replace("Story:", "Assistant:")
    return text


def convert_to_llama_chat_format(dataset):
    # Split the dataset into individual sections based on 'Features:'
    sections = dataset.split("Features:")
    conversation = ""

    for section in sections:
        if section.strip():
            # Extracting the relevant information from each section
            parts = section.split("\n")
            dialogue = parts[1].replace("Summary:", "").strip()
            story = "\n".join(parts[4:])

            # Constructing the LLAMA chat format
            conversation += f"<s>[INST] <<SYS>>\nSystem prompt: {dialogue}\n<</SYS>>\n"
            conversation += f"User prompt [/INST] {story}\nModel answer: {story}\n</s>\n"

    return conversation
