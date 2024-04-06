# --------------- IMPORTs -----------------------
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

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk

from torch.utils.tensorboard import SummaryWriter
import torch._dynamo

torch._dynamo.config.suppress_errors = True
from src.tokenize.tokenizer import Tokenizer

import os

from contextlib import nullcontext

# -------------------------------------------
BASE_URL = "/home/pranav-pc/projects/OpenTransformer/multiformer/"
TOKENIZER_CHECKPOINT = os.path.join(BASE_URL, "tokenizer_checkpoints")
tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)

block_size = 768
batch = 28  # TODO : Adjust batch size based on dynamic input dim

conf = {
    "vocab_size": 32000,
    "emebdding_dim": block_size,
    "max_seq_len": 1024,
    "embedding_dropout": 0.0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": 1.0,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "use_cache": True,
    "use_sliding_window": True,
    "residual_dropout": 0.1,
    "mlp_dropout": 0.0,
    "mlp_hidden_size": int(1.3 * 768),
    "num_layers": 5,
    "device": device,
}
ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
)
# -----------------------------------------
print('>> Data Processing')
data = load_from_disk(BASE_URL + "data/interim/TinyStories.hf")


# Define collate function to handle padding
def collate_fn(batch):
    batch = [en[:block_size] for en in batch]  # TODO : Write code for overflow
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
train_data = sorted(data["idx"], key=lambda x: len(x))

# Create DataLoader with collate function
train_loader = DataLoader(
    train_data, batch_size=batch, collate_fn=collate_fn, shuffle=False
)
# ---------------------------------------------
## Training
print('>> Training')

from src.models.blm.config import ModelArgs
from src.models.blm.model import Transformer


config = ModelArgs(**conf)
model = Transformer(config)

# Initialize TensorBoard writer for logging
writer = SummaryWriter(log_dir="tensorboard/blm/TinyStories/")

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
model.to(device)


# Training loop with profiler and tqdm
num_epochs = 3


for epoch in range(num_epochs):
    total_loss = 0

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"tensorboard//blm/TinyStories/epoch_{epoch}"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
        with_modules=False,
    ) as prof:
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)

            # logits are not being used. Let's delete it
            del logits

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Delete tensors to free up GPU memory
            del x, y, loss

    # Average loss calculation
    average_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    writer.add_scalar("Loss/Train", average_loss, epoch)

    # # Move model to CPU before saving
    # model.to('cpu')

    # Save model after every epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
    # Move model back to GPU
    # model.to(device)
    torch.cuda.empty_cache()

# Close the TensorBoard writer
writer.close()
