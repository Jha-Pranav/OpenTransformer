# OpenTransformer

From Theory to Triumph: Embracing the Application of Learnings! 📚💡

This project focuses on learning and implementing various aspects of transformer architecture, particularly tailored towards building small, efficient language models suitable for low-end consumer-grade devices. The goal is to understand and create minimalist language models with fewer than 20 million parameters, optimized for specific tasks.

## Motivation

Large language models like demand substantial hardware resources, making them unsuitable for deployment on low-end consumer-grade devices. This project aims to explore transformer architecture, develop a deep understanding, and implement efficient models that can run on resource-constrained devices.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Jha-Pranav/OpenTransformer.git
   cd multiformer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the codebase and run specific scripts for data preparation, training, fine-tuning, or inference.

## Download Model Checkpoints

I am using hf as a model-registry. You can find the list of models @ [Jha-Pranav's Hugging Face Model Hub](https://huggingface.co/Jha-Pranav/blm-lab/resolve/main/)

How to download models? Run the following command:

```bash
./download_checkpoints.sh "<name of the models separated by comma>"
```

To download the Base Language model trained on Tinystories dataset, run:

```bash
./download_checkpoints.sh "blm-medium"
```

| Model      | Description                                       | Inference Notebook                                                                 |
| ---------- | ------------------------------------------------- | ---------------------------------------------------------------------------------- |
| blm-medium | 18.6M Params Model trained on Tinystories dataset | [Text Generation Notebook](multiformer/notebooks/inference/tinystories-base.ipynb) |

## How to Fine-Tune Pretrained Models

| Task                                  | Notebook                                                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Full Finetune Causal for task         | [Full Finetune Causal](multiformer/notebooks/fine-tune/full-finetune-causal.ipynb)                  |
| Full Finetune Classification for task | [Full Finetune Classification](multiformer/notebooks/fine-tune/full-finetune-classify.ipynb)        |
| Finetune Only Last Layer              | [Finetune Only Last Layer](multiformer/notebooks/fine-tune/finetune-only-last-layer-classify.ipynb) |
| Finetune Freeze Model                 | [Fine only head](multiformer/notebooks/fine-tune/freeze-model-finetune-classify.ipynb)              |

## How to Train Model from Scratch

| Step            | Command                                                                                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Preprocess Data | `python3 src/models/blm/data_prep.py --processed_data=true --dataset_name="TinyStories-Instruct-hf" --dataset_cache_dir="TinyStories-Instruct-hf" --pack_dataset=true` |
| Create Config   | `multiformer/src/models/blm/conf/sample-config.yaml`                                                                                                                   |
| Training        | `multiformer/src/models/blm/pl_training.py`                                                                                                                            |

## License

This project is licensed under the MIT License. See the [LICENSE](multiformer/LICENSE) file for details.

## Acknowledgements

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [rasbt/substack](https://substack.com/@rasbt/posts)
- [meta-llama/llama](https://github.com/meta-llama/llama)
- [pytorch/torchtune](https://github.com/pytorch/torchtune)

Feel free to reach out with any questions, suggestions, or feedback! Let's build something amazing together! 🚀
