# Multiformer 🤖

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
