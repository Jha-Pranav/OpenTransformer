from importlib.metadata import version
import torch


print("TORCH VERSION :", version("torch"))
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backend.mps.is_available() else "cpu"
)
print("Device  : ", device.upper())

torch.manual_seed(123)
torch.cuda.manual_seed(123)

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from contextlib import nullcontext

ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
)

from model import GPT2
from config import GPT2Config
from src.cells.optim_func import config_optimizer


# ----------
num_sample = 2
max_new_tokens = 500
temperature = 0.8
top_k = 200
compile = True
profile = True
# ---------------

x = torch.randint(50304, (8, 1024), device=device)
y = torch.randint(50304, (8, 1024), device=device)
get_batch = lambda split: (x, y)

# model init
gptconf = GPT2Config(
    block_size=1024,  # how far back does the model look? i.e. context size
    n_layer=12,
    n_head=12,
    n_embd=768,  # size of the model
    dropout=0,  # for determinism
    bias=False,
)
model = GPT2(gptconf)
model.to(device)

optimizer = config_optimizer(
    model,
    weight_decay=1e-2,
    learning_rate=1e-4,
    betas=(0.9, 0.95),
    device=device,
    fused=True,
)

print("Compiling model...")
model = torch.compile(model)  # pytorch 2.0

import torch._dynamo

torch._dynamo.config.suppress_errors = True

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./bench_log"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models atm
    ) as prof:

        X, Y = get_batch("train")
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch("train")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step()  # notify the profiler at end of each step
