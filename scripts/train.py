import argparse
import torch
import random
import numpy as np

from configs.config import (
    get_baseline_config,
    get_vision_lora_config,
    get_ablation_causal_config,
)
from src.training.trainer import create_trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Vision-as-LoRA model")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["baseline", "vision_lora", "ablation_causal"],
        required=True,
        help="Experiment configuration to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    if args.experiment == "baseline":
        config = get_baseline_config()
    elif args.experiment == "vision_lora":
        config = get_vision_lora_config()
    elif args.experiment == "ablation_causal":
        config = get_ablation_causal_config()
    
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    
    set_seed(config.training.seed)
    
    print("\n" + "="*60)
    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print("="*60 + "\n")
    
    trainer = create_trainer(config)
    
    trainer.train()
    
    print("\nTraining completed successfully")


if __name__ == "__main__":
    main()