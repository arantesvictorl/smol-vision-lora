import argparse
import torch
from torch.utils.data import DataLoader

from configs.config import Config, get_vision_lora_config
from src.model.vision_lora_model import VisionLoRAModel
from src.data.dataset import VisionLanguageDataset
from src.training.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum batches for evaluation"
    )
    
    args = parser.parse_args()
    
    config = get_vision_lora_config()
    
    print(f"Loading model from {args.checkpoint}")
    model = VisionLoRAModel(config)
    
    print("Loading validation dataset")
    val_dataset = VisionLanguageDataset(config, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
    )
    
    print("Running evaluation")
    metrics = evaluate_model(model, config, val_loader)
    
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()