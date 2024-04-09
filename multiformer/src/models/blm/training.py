# --------------- IMPORTs -----------------------
from importlib.metadata import version

print("TORCH VERSION :", version("torch"))


# Silence all warnings
import warnings

warnings.filterwarnings("ignore")

import torch

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

from torch.utils.tensorboard import SummaryWriter
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from contextlib import nullcontext

from src.models.blm.create_dataloader import data_iter

# -------------------------------------------
batch = 64
block_size = 1024
# --------------------------------------

train_loader = data_iter(batch=batch, block_size=block_size)

conf = {
    "vocab_size": 32000,
    "emebdding_dim": 512,
    "max_seq_len": block_size,
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
## Training
print(">> Training")

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
    fused=False,
)
model.to(device)
model = torch.compile(model, dynamic=True)
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
            f"tensorboard/blm/TinyStories/epoch_{epoch}"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
        with_modules=False,
    ) as prof:
        for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            with ctx:
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

    # Save model after every epoch
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

    torch.cuda.empty_cache()

# Close the TensorBoard writer
writer.close()
