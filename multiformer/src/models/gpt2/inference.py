from importlib.metadata import version
import torch
import tiktoken

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

# ----------
num_sample = 2
max_new_tokens = 500
temperature = 0.8
top_k = 200
compile = True
# ---------------


model = GPT2.from_pretrained("gpt2", dict(dropout=0.0))
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)


enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


s = "What is the meaning of name 'Pranav'"
idx = encode(s)

x = torch.tensor(idx, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_sample):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print("---------------" * 10)
