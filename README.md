# Unified Multimodal Understanding via Byte-Pair Visual Encoding

<p align="center">
    <img src="docs/static/images/being-vl-05.png" width="300"/>
<p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Website-Being--VL--0.5-green)](https://beingbeyond.github.io/Being-VL-0.5)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23639-b31b1b.svg)](https://arxiv.org/abs/2506.23639)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>

<p align="center">
    <img src="docs/static/images/framework.png" width="600"/>
<p>

**Being-VL-0.5** is an MLLM that combines text and image understanding using a novel approach called Visual Byte-Pair Encoding (vBPE). Instead of treating images and text as completely separate modalities, our method applies BPE tokenization directly to visual tokens, creating a more unified representation that helps the model better understand relationships between vision and language.

For more details, please refer to our paper: [Unified Multimodal Understanding via Byte-Pair Visual Encoding](https://arxiv.org/abs/2506.23639) (**ICCV'25**).

## News

- **[2025-07-12]**: 🔥🔥 We release the code and training scripts!
- **[2025-06-26]**: 🎉🎉 We publish **Being-VL-0.5**, which is accepted by **ICCV 2025**! Check our paper [here](https://arxiv.org/abs/2506.23639). The code and training scripts will be released soon.
- **[2025-01-23]**: 🎉🎉 **Being-VL-0** is accepted by ICLR 2025! Check our paper [here](https://openreview.net/pdf?id=3TnLGGHhNx).
- **[2024-10-03]**: We publish **Being-VL-0**, the first version of **Being-VL** series.


## Quick Start

### 1. Installation

```bash
pip install -e .
pip install flash-attn --no-build-isolation
pip install -e ./transformers
```

*We have made some modifications to the transformers library to support our model. Please install our provided transformers folder.*

### 2. Directory Setup

Create a workspace with the following structure:

```
/path/to/your/workspace/
├── models/
│   ├── llama3.1-8b/              # Base LLaMA model
│   ├── being-vq-8k/              # Being VQ-GAN model
│   ├── being-tokenizer/          # Being tokenizer
│   └── beingvl/                  # Output models
│       ├── base/                 # Initialized model
│       ├── stage-1/              # Stage 1 output
│       ├── stage-2/              # Stage 2 output
│       └── stage-3/              # Final model
├── data/
│   ├── images/                   # Raw images
│   ├── annotations/              # JSON annotations
│   ├── vq_tokens/                # VQ encoded tokens (.npy)
│   ├── vbpe/                     # vBPE tokenizer (.pkl)
│   └── tokenized/                # Tokenized datasets (.jsonl)
│       ├── pt/                   # Stage 1 PT data
│       ├── sft_stage2/           # Stage 2 SFT data
│       └── sft_stage3/           # Stage 3 SFT data
└── logs/                         # Training logs
    ├── stage-1/
    ├── stage-2/
    └── stage-3/
```

### 3. Base Model Initialization

**Requirements:**
- Downloaded [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) checkpoint. You can also use any other text-LLM, but it will require additional configuration (eg, dimensions, processing codes, etc).
- Pretrained [VQ-GAN](https://huggingface.co/zawnpn/being-vq-8k) checkpoint. This is extracted from Meta's [Chameleon](https://huggingface.co/facebook/chameleon-7b) weights and converted to adapt to Being-VL. You can also use your own VQ-GAN models.
- Being tokenizer config: [beingvl/config/being-tokenizer-config](beingvl/config/being-tokenizer-config)

Initialize the Being-VL base model from Llama-3.1-8B using the provided tokenizer configuration:

```bash
# Download Llama-3.1-8B model (if not already available)
# Place it in /path/to/your/workspace/models/llama3.1-8b/

# Initialize Being-VL base model
python beingvl/utils/convert_llama_beingvl.py \
    --llama_path /path/to/your/workspace/models/llama3.1-8b \
    --being_tokenizer_config_path beingvl/config/being-tokenizer-config \
    --being_vq_path /path/to/your/workspace/models/being-vq-8k \
    --output_path /path/to/your/workspace/models/beingvl/base \
    --verify_loading
```

This creates and initializes a Being-VL base model with extended vocabulary for VQ and vBPE tokens, ready for 3-stage training.

### 4. Training Pipeline

Being-VL uses a 3-stage training methodology:

```bash
# Stage 1
bash beingvl/scripts/train-stage-1.sh 0

# Stage 2
bash beingvl/scripts/train-stage-2.sh 0

# Stage 3
bash beingvl/scripts/train-stage-3.sh 0
```

### 5. Documentation

For detailed instructions, see:

- **[Data.md](docs/Data.md)**: Data preparation and VQ encoding
- **[Train.md](docs/Train.md)**: Training configuration and commands
- **[Inference.md](docs/Inference.md)**: Using the trained model for inference

## Acknowledgements
We thank the open-sourcing projects [Chameleon](https://github.com/facebookresearch/chameleon) and [Transformers](https://github.com/huggingface/transformers), as our code is developed based on them.

## Disclaimer
The code has been refactored from our development version and may not be fully tested in the new codebase. Some minor functions are still in progress and will be included later. If you encounter any issues, please open an issue to help us improve the project.

## Citation
If you find our work useful, please consider citing us and give a star to our repository! 🌟🌟🌟

**Being-VL-0.5**

```bibtex
@inproceedings{zhang2025beingvl05,
  title={Unified Multimodal Understanding via Byte-Pair Visual Encoding},
  author={Zhang, Wanpeng and Feng, Yicheng and Luo, Hao and Li, Yijiang and Yue, Zihao and Zheng, Sipeng and Lu, Zongqing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

**Being-VL-0**

```bibtex
@inproceedings{zhang2025beingvl0,
  title={From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities},
  author={Zhang, Wanpeng and Xie, Zilong and Feng, Yicheng and Li, Yijiang and Xing, Xingrun and Zheng, Sipeng and Lu, Zongqing},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=3TnLGGHhNx}
}
```