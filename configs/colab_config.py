from dataclasses import dataclass
from configs.config import Config, ModelConfig, VisionConfig, LoRAConfig, TrainingConfig, DataConfig, ExperimentConfig


def get_colab_test_config() -> Config:
    """
    Configuration optimized for Colab A100 testing.
    
    Reduced scale for quick validation before full H100 training.
    """
    return Config(
        model=ModelConfig(
            model_name="HuggingFaceTB/SmolLM2-135M",
            torch_dtype="bfloat16",
            use_flash_attention=True,
            device_map="cuda",
        ),
        vision=VisionConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=576,
        ),
        lora=LoRAConfig(
            rank=64,
            alpha=128,
            dropout=0.05,
            vision_layers=8,
        ),
        data=DataConfig(
            dataset_name="nielsr/coco-captions",
            train_split="train[:1%]",
            val_split="validation[:1%]",
            num_workers=2,
            prefetch_factor=2,
        ),
        training=TrainingConfig(
            output_dir="/content/drive/MyDrive/vision-lora-test",
            run_name="colab-a100-test",
            batch_size=16,
            gradient_accumulation_steps=2,
            max_length=128,
            target_samples=1000,
            learning_rate=5e-4,
            warmup_steps=10,
            logging_steps=5,
            eval_steps=50,
            save_total_limit=1,
            use_bf16=True,
            use_torch_compile=False,
        ),
        experiment=ExperimentConfig(
            name="colab_test",
            use_vision=True,
            use_bidirectional_mask=True,
            description="Quick validation test on Colab A100"
        ),
        wandb_project="vision-lora-colab-test",
    )


def get_colab_full_config() -> Config:
    """
    Configuration for extended testing on Colab A100.
    
    Scaled to use full Colab session (12h max).
    """
    return Config(
        model=ModelConfig(
            model_name="HuggingFaceTB/SmolLM2-135M",
            torch_dtype="bfloat16",
            use_flash_attention=True,
        ),
        vision=VisionConfig(
            image_size=224,
            patch_size=16,
        ),
        lora=LoRAConfig(
            rank=128,
            alpha=256,
            vision_layers=12,
        ),
        data=DataConfig(
            dataset_name="nielsr/coco-captions",
            train_split="train[:25%]",
            val_split="validation[:10%]",
            num_workers=4,
        ),
        training=TrainingConfig(
            output_dir="/content/drive/MyDrive/vision-lora-full",
            run_name="colab-a100-full",
            batch_size=24,
            gradient_accumulation_steps=4,
            max_length=256,
            target_samples=250_000,
            learning_rate=5e-4,
            warmup_steps=100,
            logging_steps=25,
            eval_steps=500,
            save_total_limit=2,
            use_bf16=True,
            use_torch_compile=False,
        ),
        experiment=ExperimentConfig(
            name="colab_full",
            use_vision=True,
            use_bidirectional_mask=True,
            description="Extended training on Colab A100 (12h)"
        ),
        wandb_project="vision-lora-colab",
    )