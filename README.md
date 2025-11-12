<div align="center">
    
# 🎭 Masked Diffusion Language Models <br> with Frequency-Informed Training

### ⭐ Winners of the Strict Track (NLP tasks) of the BabyLM Challenge 2025
###             Oral presentation - BabyLM Workshop @ EMNLP 2026

[**📄 Paper**](https://arxiv.org/abs/2509.05056) • [**🤗 Models**](https://huggingface.co/despoinakk) • [**💻 Code**](https://github.com/DespoinaKK/babylm-diffusion)

</div>

---

Official PyTorch implementation of **"Masked Diffusion Language Models with Frequency-Informed Training"**

## 🌟 Overview

This repository implements a masked diffusion language modeling framework for data-efficient training under strict data constraints. Our approach combines:

- **Masked Diffusion Language Models (MDLMs)**: A discrete diffusion approach that uses masked language modelling (bidirectional context) to generate text
- **Novel Noise Schedules**: Bimodal Gaussian (our best model) and cosine (submission's model)
- **Frequency-Informed Masking**: Progressive prioritization (curriculm) of rare tokens during training, which integrates seamlessly with the MDLM framework
- **NELBO Reweighting**: Exploration of different weighting schemes to optimize performance across schedules

![Noise Schedule Evolution](assets/noise_schedule_evolution.gif)


### 🎯 Results
Our method achieves competitive performance with state-of-the-art baselines (GPT-BERT) on the BabyLM benchmarks, demonstrating that diffusion-based training offers a viable alternative for data-restricted language learning.

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/DespoinaKK/babylm-diffusion
cd babylm-diffusion

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Data Preparation
For our models, we use the BabyLM corpus, for the strict track (100M words). You can find more information [here](https://babylm.github.io/).

First, train a BPE tokenizer on your corpus:
```bash
python tokenization/create_tokenizer.py \
    --input_path /path/to/data \
    --vocab_size 16384 \
    --vocab_path ./tokenizers/tokenizer.json
```

Then tokenize your dataset:
```bash
python tokenization/tokenize_corpus.py \
    --data_folder /path/to/data \
    --train_file data.train \
    --valid_file data.valid \
    --tokenizer_folder ./tokenizers \
    --tokenizer_file tokenizer.json \
    --name tokenized
```

### 2. Training

For distributed training, adapt the scripts in `slurm-scripts`. Our models are trained on a single node with 4 A100 64GB gpus (you can adapt the setup to utilize only 1 gpu, modifying the slurm*.sh scripts). To specify the hyperparameters and noise schedules you want, modify the config files in `slurm-scripts`.

## 🔑 Key Components

### 🌊 Noise Schedules

**Cosine Schedule** (`masking/noise_schedules.py`):
- Focuses on lower masking rates
- Average masking rate: 0.36

**Gaussian Schedules**:
- **Unimodal**
- **Bimodal**: Mixture distribution combining low and high masking modes
- Requires derivative softening (γ < 1.0) for stable training


### 📊 Frequency-Informed Masking

The frequency-informed masking strategy assigns higher masking probabilities to rare tokens. Implementation in `masking/frequency_masking.py`:

1. **Token Ranking**: Tokens ranked by global corpus frequency
2. **Min-Max Normalization**: Per-sequence normalization to [0,1]
3. **Softening**: Weights raised to power p < 1 to prevent over-emphasis on extremely rare tokens
4. **Conditional Scaling**: Ensures mean masking probability matches target rate (1 - α_t)

Optional curriculum learning progressively increases p from 0 to 0.02 across epochs.


## 📈 Evaluation

Models are evaluated using the [BabyLM Challenge evaluation pipeline](https://github.com/babylm/evaluation-pipeline-2025/) with MLM pseudo-likelihood backend:

- **Zero-shot**: BLiMP, BLiMP Supplement, EWoK, Entity Tracking, COMPS
- **Finetuning**: GLUE and SuperGLUE subsets. For finetuning, we provide the `eval-utils/classifier.py` helper file, with minor tensor shape changes to match our models. 
- **Human-likeness**: Reading task, Age of Acquisition, morphology tasks correlation with human performance

Evaluation can be run with or without time conditioning (see paper Section 3.2).


## 🎯 Updated Results - Bimodal Gaussian Schedule
See paper for full results and ablations.
We also include our new 512-seq-len model's results, trained with the Gaussian Bimodal noise schedule described in the paper:

Performance comparison on BabyLM Challenge zero-shot tasks:

| Task | Baseline <br> (GPT-BERT)  | Submission <br> (Cosine) | **Best <br> (Bimodal Gaussian)** |
|------|-------------------|---------------------|-------------------------------|
| BLiMP | 80.5 | 76.9 | 78.2 |
| BLiMP Supplement | 73.0 | 72.4 |73.6 |
| EWoK | 52.4 | 51.8 | 52.5 |
| COMPS | 59.7 | 56.4 | 56.6 |
| Entity Tracking | 39.9 | 40.8 | 39.7 |

## 🤗 Hugginface Pretrained Models

| Model | Downloads |
|-------|-----------|
| [Cosine - Submission](https://huggingface.co/despoinakk/diffusion_cosine_babylm) | ![](https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/models/despoinakk/diffusion_cosine_babylm&query=$.downloads&label=&color=green) |
| [Bimodal Gaussian - Best](https://huggingface.co/despoinakk/diffusion_gaussian_babylm) | ![](https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/models/despoinakk/diffusion_gaussian_babylm&query=$.downloads&label=&color=green) |


## 📝 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{kosmopoulou2025masked,
  title={Masked Diffusion Language Models with Frequency-Informed Training},
  author={Kosmopoulou, Despoina and Georgiou, Efthymios and Dorovatas, Vaggelis and Paraskevopoulos, Georgios and Potamianos, Alexandros},
  journal={arXiv preprint arXiv:2509.05056},
  year={2025}
}
```

## 📚 References

This repo is based on work from:
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [GPT or BERT: why not both?](https://aclanthology.org/2024.conll-babylm.23/) 
- Architecture adapted from [LTG-BERT](https://arxiv.org/abs/2303.09859)


## 📁 Code Structure
```
.
├── main.py                      # Main training entry point
├── model.py                     # Transformer model with diffusion
├── config/                      # Configuration and arguments
│   ├── arguments.py            # CLI argument parsing
│   └── model_configuration.py  # Model architecture config
├── data/                        # Data loading and processing
│   ├── dataset.py              # Dataset implementation
│   ├── dataset_manager.py      # Manage Datasets (loading)
│   └── dataset_utils.py        # Utilities for data
├── eval-utils
│   └── classifier_model.py      # file to use for finetuning
├── masking/                     # Masking strategies
│   ├── noise_schedules.py      # Diffusion noise schedules
│   ├── masking_processor.py    # Token masking logic
│   ├── frequency_masking.py    # Frequency-informed masking
│   └── batch_processing.py     # Batch preparation
├── training/                    # Training infrastructure
│   ├── training_loop.py        # Main training loop
│   ├── ema.py                  # Exponential moving average
│   ├── checkpoint_manager.py   # Checkpoint saving
│   ├── validation.py           # Validation during training
│   ├── model_setup.py          # Model loading and optimizer setup
│   └── distributed_setup.py    # DDP setup
├── tokenization/                # Tokenization scripts
│   ├── create_tokenizer.py     # Train BPE tokenizer
│   └── tokenize_corpus.py      # Tokenize datasets
├── optimization/                # Optimizers
│   └── lamb.py                 # LAMB optimizer
└── slurm-scripts/               # Scripts
    ├── slurm-train.sh          # SLURM job script
    ├── launch-train.sh         # Local launch script
    ├── config-cosine.json      # Cosine schedule config example 
    └── config-gauss.json       # Gaussian schedule config example
```



## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- [Despoina](https://scholar.google.com/citations?user=roxd-tsAAAAJ&hl=en&oi=sra) | [github](https://github.com/DespoinaKK) | despoinakkosmopoulou[at]gmail[dot]com
- [Efthymis](https://scholar.google.com/citations?user=5Sc6GvEAAAAJ&hl=en) | [github](https://github.com/efthymisgeo) | efthymios[dot]georgiou[at]unibe[dot]ch
