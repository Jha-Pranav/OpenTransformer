from importlib.metadata import version
import torch
import warnings

# Silence all warnings
warnings.filterwarnings("ignore")

print("TORCH VERSION :", version("torch"))
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backend.mps.is_available() else "cpu"
)
print("GPU  : ", device.upper())

torch.manual_seed(123)
torch.cuda.manual_seed(123)

import datasets

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from src.tokenize.tokenizer import Tokenizer
#-------------------------------------------
block_size  = 1024
batch = 8   # TODO : Adjust batch size based on dynamic input dim
#-----------------------------------------



TOKENIZER_CHECKPOINT = (
    "/home/pranav-pc/projects/OpenTransformer/multiformer/tokenizer_checkpoints/"
)

# Load dataset from Hugging Face datasets library
from datasets import load_dataset

dataset = load_dataset("imdb")

tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)

data = dataset.map(
    lambda example: {"idx": [en[:block_size] for en in tokenizer.encode(example["text"])]},
    batch_size=512,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# Define collate function to handle padding
def collate_fn(batch):
    x_batch = [torch.tensor(en[:-1]) for en in batch]  # Extract x (remove last token)
    y_batch = [torch.tensor(en[1:]) for en in batch]  # Extract y (remove first token)
    x_padded = pad_sequence(
        x_batch, batch_first=True, padding_value=tokenizer.eos_id()
    )  # Pad x sequences
    y_padded = pad_sequence(
        y_batch, batch_first=True, padding_value=tokenizer.eos_id()
    )  # Pad y sequences
    return x_padded, y_padded


# Sort the data and turn off shuffle - Simplest way of implementing Seq leng batch sampling
train_data = sorted(data["train"]["idx"], key=lambda x: len(x))

# Create DataLoader with collate function
train_loader = DataLoader(
    train_data, batch_size=batch, collate_fn=collate_fn, shuffle=False
)


## Training

from src.models.gpt2.config import GPT2Config
from src.models.gpt2.model import GPT2

gpt_conf = {
    "block_size": block_size,
    "vocab_size": tokenizer.vocab_size(),
    "n_layer": 8,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "bias": True,
    "device": device,
}

config = GPT2Config(**gpt_conf)
model = GPT2(config)


from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer for logging
writer = SummaryWriter(log_dir="gpt2_log")

from src.cells.optim_func import configure_optimizers

optimizer = configure_optimizers(
    model,
    weight_decay=1e-2,
    learning_rate=1e-4,
    betas=(0.9, 0.95),
    device=device,
    fused=True,
)
model.to(device)
model = torch.compile(model)

from contextlib import nullcontext

ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
)

# Training loop
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(num_epochs):
    total_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        # Perform forward pass on GPU
        logits, loss = model(x, y)

        # logits are not being used. let's delete it 

        # Compute gradients on CPU
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./gpt2_log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        ) as prof:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

        # Update model parameters on GPU
        optimizer.step()
        total_loss += loss.item() * x.size(0)

        if (batch_idx + 1) % 2000 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Iteration [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )
            checkpoint_path = f"model_checkpoint_epoch_{epoch+1}_iter_{batch_idx+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at {checkpoint_path}")

        # Delete tensors to free up GPU memory
        del x, y, logits, loss

    # Average loss calculation
    average_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")


    # # Save model after every epoch
    # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

# Close the TensorBoard writer
writer.close()
