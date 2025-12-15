# Bayesian Low-Rank Adaptation for Generative Tasks (BLoB Extension)

**Course Project – Deep Learning, Advanced Course (DD2610)**  
**Authors:**  
- Fredrik Ström - frest@kth.se
- William Rosengren - wrose@kth.se
- Yuusuf Dahlstrand - yuusufd@kth.se

This repository is an **extension of the original Bayesian PEFT (BLoB) codebase** by Wang et al. (NeurIPS 2024), adapted and extended as part of a course project.

Our work focuses on reproducing, extending, and analyzing Bayesian Low-Rank Adaptation (BLoB) in encoder–decoder architectures and generative tasks, with a particular emphasis on abstractive summarization and uncertainty analysis.

---

## Overview of the Original Codebase

This repository builds on the official implementation of:

- **BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models**  
  Y. Wang*, H. Shi*, L. Han, D. Metaxas, H. Wang  
  *NeurIPS 2024*  
  [[Paper]](https://arxiv.org/abs/2406.11675)

The original repository provides:
- Bayesian and deterministic LoRA training
- Classification-focused experiments (encoder-only models)
- Evaluation on GLUE-style benchmarks

**Original repository: https://github.com/Wang-ML-Lab/bayesian-peft**

---

## Extensions Introduced in This Repository

This project extends BLoB beyond its original scope in the following ways:

### 1. Generative Tasks via Encoder–Decoder Models
- Added **`blob_summarization`** wrapper to support sequence-to-sequence models
- Applied BLoB to **BART-base** for abstractive summarization
- Inserted Bayesian LoRA adapters into both encoder and decoder projections

### 2. Hyperparameter Sensitivity Analysis
- New scripts for systematic sweeps over:
  - β (KL weight)
  - γ (prior scaling)
- Analysis focuses on:
  - KL collapse
  - ELBO behavior
  - Stability of Bayesian training

### 3. Monte Carlo Inference for Text Generation
- Implemented Bayesian inference at generation time by sampling adapter weights
- Generated multiple summaries per input using deterministic decoding
- Enabled empirical analysis of output variability induced solely by weight uncertainty

### 4. Large-Scale Sampling Infrastructure (Modal)
- Added **`modal_blob.py`** to run long-running Monte Carlo inference on GPUs
- Supports:
  - Resumable execution
  - Safe JSON checkpointing
  - Large-scale sampling (e.g., 100 dialogues × 100 posterior samples)

---

## Repository Structure (Key Additions)

```text
.
── models/
│   └── seq2seq.py              # seq2seq 
├── modelwrappers/
│   └── blob_summarization.py   # BLoB ext. for sum. task.
├── modal_blob.py               # NEW: GPU sampling + MC inference
├── scripts/
│   ├── blob/                   # extended (BART scripts)
├── utils/
│   ├── args.py                 # extended (added new arguments for generative compatibility)
├── blob_mc_samples/            # generated MC summaries (JSON)
└── README.md                   # this file
```

## 📚 References
[BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models](https://arxiv.org/abs/2406.11675)
```bib
@article{wang2024blob,
  title={BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models},
  author={Wang, Yibin and Shi, Haizhou and Han, Ligong and Metaxas, Dimitris and Wang, Hao},
  journal={arXiv preprint arXiv:2406.11675},
  year={2024}
}
```