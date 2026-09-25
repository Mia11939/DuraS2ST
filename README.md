<div align="center">
  <h1>DuraS2ST</h1>
  <p><b><i>Chain-of-Thought and Reinforcement Learning for Duration-Aligned Speech-to-Speech Translation</i></b></p>
  <p>Official repository of the EMNLP 2026 Main Conference paper.</p>
  <p>
    <img src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b.svg?logo=arxiv&logoColor=white" alt="Paper coming soon">
    <img src="https://img.shields.io/badge/Model-Coming%20Soon-FFD21E.svg?logo=huggingface&logoColor=black" alt="Model coming soon">
    <img src="https://img.shields.io/badge/Dataset-Coming%20Soon-2F80ED.svg?logo=huggingface&logoColor=white" alt="Dataset coming soon">
    <a href="https://huggingface.co/spaces/Mia11939/DuraS2ST-Demo"><img src="https://img.shields.io/badge/Audio%20Demo-Hugging%20Face-FF9D00.svg?logo=huggingface&logoColor=white" alt="Audio demo"></a>
  </p>
</div>

---

## News

- **[2026-08-21]** 🎉 **DuraS2ST** is accepted to the **EMNLP 2026 Main Conference** (**15.4% acceptance rate** for Main Conference papers).

## Introduction

**DuraS2ST** is a reasoning-based framework for duration-aligned speech-to-speech translation. It explicitly plans the target wording and phonetic length before speech generation, and is optimized with duration-aware multimodal reinforcement learning.

This repository supports English↔Chinese speech translation with vLLM.

## Overview

<div align="center">
  <img src="assets/framework.png" alt="DuraS2ST framework" width="98%">
</div>

## Installation

Requires Linux (x86_64), FFmpeg, and an NVIDIA GPU with a CUDA 12.8-compatible driver
(tested on an A100 80 GB). Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/Mia11939/DuraS2ST.git
cd DuraS2ST
uv sync --frozen
```

The locked environment includes the StepAudio2 vLLM backend and speech decoder,
sharing the same PyTorch installation. CUDA kernels are downloaded prebuilt.

## Inference

Download the model, including its `token2wav/` directory, then run:

```bash
uv run --frozen inference.py \
  --model /path/to/DuraS2ST_model \
  --input-audio /path/to/source.wav \
  --target-language zh \
  --output-audio outputs/translation.wav
```

Use `zh` for English-to-Chinese and `en` for Chinese-to-English. The script
prints the translation and saves the generated speech. Add `--show-reasoning`
to print the duration-planning rationale.

`--model` accepts a local directory or a Hugging Face model ID.
The input recording also supplies the speaker prompt.

## Acknowledgements

This project is built on [Step-Audio 2](https://github.com/stepfun-ai/Step-Audio2). We thank the authors for releasing their models and inference code.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). See the model repository for its license. Third-party attribution is retained in [NOTICE](NOTICE).
