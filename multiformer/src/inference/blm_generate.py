import torch
from src.models.blm.pl_training import Transformer
from src.tokenize.tokenizer import Tokenizer

MODEL_CHECKPOINT_PATH = "/home/pranav-pc/projects/model-registry/blm-medium/last.ckpt"
TOKENIZER_CHECKPOINT = (
    "/home/pranav-pc/projects/OpenTransformer/multiformer/tokenizer_checkpoints/"
)
tokenizer = Tokenizer(TOKENIZER_CHECKPOINT)
model = Transformer.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
model = torch.compile(model, dynamic=True)

#### Inference
model.eval()
model = model.cuda()

import os

os.environ["WANDB_DISABLED"] = "true"


def predict(text, max_new_tokens=1024, temperature=0.8, top_k=3, conditional_break=[13, 13, 1]):
    tokens = torch.LongTensor(tokenizer.encode_as_ids(text))[:-1].to("cuda:0").view(1, -1)
    # print('tokens',ds.tokenizer.encode(text,out_type=str)[:-1])
    print(
        tokenizer.decode_ids(
            model.predict_step(
                tokens,
                None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                conditional_break=conditional_break,
            )[0].tolist()
        )
    )
