import ChatTTS
import numpy as np
import torch
import torchaudio
from IPython.display import Audio

with open("conversation-sample2.txt") as file:
    content = file.read()

import re


# Function to clean the text
def clean_text(text):
    # Remove timestamps
    text = re.sub(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ", "", text)

    # Remove labels (ASSISTANT, USER, etc.)
    text = re.sub(r"\b(ASSISTANT|USER):\s?", "", text)

    # Remove special characters except periods
    text = re.sub(r"[^\w\s.]", "", text)

    return text.strip()


# Function to parse logs and return a list of dialogues
def parse_logs(log_data):
    # Split the log by newlines to separate each line
    log_lines = log_data.split("\n")

    # Parse each line and clean it
    dialogues = [clean_text(line) for line in log_lines if line.strip()]

    return dialogues


# Parse the log data
dialogues = parse_logs(content)

# Initialize the ChatTTS model
chat = ChatTTS.Chat()
chat.load(compile=False, source="local", custom_path="/home/pranav-pc/projects/angel/")
# Generate speech from the text

###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()


params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.3,  # using custom temperature
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7)
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt="[oral_2][laugh_0][break_6]",
)

wavs = chat.infer(
    dialogues,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)


# Function to trim silence (values close to zero) from the end of audio
def trim_silence(audio, threshold=1e-3):
    """
    Trims silence from the end of the audio.
    Args:
        audio (numpy array): The audio data.
        threshold (float): Values below this threshold are considered silence.
    Returns:
        numpy array: Trimmed audio.
    """
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent_indices) == 0:  # If all is silence
        return audio
    return audio[: non_silent_indices[-1] + 1]


# Trim silence from the end of each audio segment
trimmed_wavs = [trim_silence(wav) for wav in wavs]

# Merge all the audio into one tensor (along dimension 0 for 1D tensors)
merged_audio = torch.cat([torch.from_numpy(wav).squeeze() for wav in trimmed_wavs], dim=0)

# Play the merged audio directly in the notebook
# display(Audio(merged_audio.numpy(), rate=24000))  # You can adjust the rate if needed

# """
# In some versions of torchaudio, the first line works but in other versions, so does the second line.
# """
try:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)
