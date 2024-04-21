# Multiformer 🤖

This project focuses on learning and implementing various aspects of transformer architecture, particularly tailored towards building small, efficient language models suitable for low-end consumer-grade devices. The goal is to understand and create minimalist language models with fewer than 20 million parameters, optimized for specific tasks.

## Motivation

Large language models like demand substantial hardware resources, making them unsuitable for deployment on low-end consumer grade devices. This project aims to explore transformer architecture, develop a deep understanding, and implement efficient models that can run on resource-constrained devices.

## Libraries Used

- pytorch
- pytorch-lightning
- sentencepiece-tokenizer

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Jha-Pranav/OpenTransformer.git
   cd multiformer
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

3. Explore the codebase and run specific scripts for data preparation, training, fine-tuning, or inference.

## Project Structure

| Folder/File       | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| `cells`           | Contains modules for different components of the transformer architecture. |
| `data_wrangling`  | Includes scripts for data preprocessing and manipulation.                  |
| `models`          | Holds implementations of different models, including BLM and GPT2.         |
| `tokenize`        | Contains files related to tokenization of text data.                       |
| `weights_adapter` | Includes scripts for adapting weights between different models.            |

### Experimental Notebooks Overview

| Folder/File           | Description                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `data-prep`           | Notebooks for data preparation tasks, including tokenization, de-duplication, and data stream processing. |
| `eval`                | Notebooks for evaluating model performance and metrics, along with datasets for evaluation.               |
| `experiments`         | Notebooks for conducting various experiments, such as implementing a VAE on embeddings.                   |
| `fine-tune`           | Notebooks for fine-tuning models on specific tasks, including causal and non-causal approaches.           |
| `inference`           | Notebooks for inference tasks, including generating text using trained models.                            |
| `interpretability`    | (Folder) Notebooks related to model interpretability techniques.                                          |
| `lora`                | Notebooks related to the LoRa project.                                                                    |
| `models`              | Notebooks specific to model implementations and training, such as BLM and GPT2.                           |
| `optimization`        | Notebooks focusing on optimization techniques, such as performance tuning and torch compilation.          |
| `Q`                   | Notebooks related to quantization techniques.                                                             |
| `transformer_anatomy` | Notebooks exploring the anatomy of transformer models, covering various components and mechanisms.        |
| `weights_adapter`     | Notebooks for adapting weights between different models, including Hugging Face and others.               |

Each notebook serves a specific purpose, ranging from data preparation and model training to evaluation and optimization techniques.

## License

This project is licensed under the MIT License. See the [LICENSE](multiformer/LICENSE) file for details.

## Acknowledgements

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [rasbt/MachineLearning-QandAI-book](https://github.com/rasbt/MachineLearning-QandAI-book)
- [meta-llama/llama](https://github.com/meta-llama/llama)
- [pytorch/torchtune](https://github.com/pytorch/torchtune)

Feel free to reach out with any questions, suggestions, or feedback! Let's build something amazing together! 🚀
