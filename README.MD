# Vision as LoRA: Integrating Vision Capabilities into Small Language Models

Implementation of Vision-as-LoRA for integrating vision capabilities into SmolLM2 using Low-Rank Adaptation.

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
vision-as-lora/
├── configs/          # Configuration files
├── src/              # Source code
│   ├── model/        # Model implementations
│   ├── data/         # Dataset implementations
│   └── training/     # Training and evaluation
├── scripts/          # Executable scripts
└── outputs/          # Training outputs
```

## Quick Start

### Training

Run the main Vision-as-LoRA experiment:
```bash
python scripts/train.py --experiment vision_lora
```

Run baseline (text-only):
```bash
python scripts/train.py --experiment baseline
```

Run ablation study (causal masking):
```bash
python scripts/train.py --experiment ablation_causal
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint outputs/final
```

### Inference
```bash
python scripts/inference.py \
    --checkpoint outputs/final \
    --image path/to/image.jpg
```

## Experiments

### 1. Baseline (Text-Only)
- SmolLM2-135M without vision
- ~100k samples, ~2 hours training

### 2. Vision-as-LoRA (Main)
- SmolLM2-135M + Vision LoRA
- Bidirectional attention for vision tokens
- ~400k samples, ~8 hours training

### 3. Ablation Study
- Vision-as-LoRA with causal masking
- ~100k samples, ~2 hours training

## Configuration

All configurations are defined in `configs/config.py` using dataclasses.

Key parameters:
- Model: SmolLM2-135M
- LoRA rank: 128
- Vision layers: 12
- Image size: 224x224
- Batch size: 32 (effective: 128)

## Citation
```bibtex
@article{wang2025visionaslora,
  title={Vision as LoRA},
  author={Wang, Han and others},
  journal={arXiv preprint arXiv:2503.20680},
  year={2025}
}
```