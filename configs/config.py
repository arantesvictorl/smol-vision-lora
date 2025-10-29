from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    device_map: str = "cuda"
    trust_remote_code: bool = True


@dataclass
class VisionConfig:
    """Configuration for vision components."""
    
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_dim: int = 576
    max_patches: int = 196
    
    def __post_init__(self):
        self.max_patches = (self.image_size // self.patch_size) ** 2


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation."""
    
    rank: int = 128
    alpha: int = 256
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"
    ])
    bias: str = "none"
    vision_layers: int = 12
    
    @property
    def lora_alpha(self) -> int:
        return self.alpha


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    
    dataset_name: str = "nielsr/coco-captions"
    train_split: str = "train[:50%]"
    val_split: str = "validation[:10%]"
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    output_dir: str = "./outputs"
    run_name: str = "smolvision-12h"
    
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    
    target_samples: int = 400_000
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    use_bf16: bool = True
    use_torch_compile: bool = True
    
    logging_steps: int = 50
    eval_steps: int = 500
    save_total_limit: int = 3
    
    seed: int = 42
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def max_steps(self) -> int:
        return self.target_samples // self.effective_batch_size


@dataclass
class ExperimentConfig:
    """Configuration for experiment variants."""
    
    name: str = "vision_as_lora"
    use_bidirectional_mask: bool = True
    use_vision: bool = True
    description: str = ""


@dataclass
class Config:
    """Main configuration aggregating all components."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    wandb_project: str = "vision-as-lora-tcc"
    wandb_entity: Optional[str] = None
    
    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "vision": self.vision.__dict__,
            "lora": self.lora.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "experiment": self.experiment.__dict__,
        }


def get_baseline_config() -> Config:
    """Configuration for baseline experiment (text-only)."""
    config = Config()
    config.experiment = ExperimentConfig(
        name="baseline_text_only",
        use_vision=False,
        use_bidirectional_mask=False,
        description="Baseline SmolLM2-135M without vision capabilities"
    )
    config.training.target_samples = 100_000
    config.training.run_name = "baseline-text-only"
    return config


def get_vision_lora_config() -> Config:
    """Configuration for main Vision-as-LoRA experiment."""
    config = Config()
    config.experiment = ExperimentConfig(
        name="vision_as_lora",
        use_vision=True,
        use_bidirectional_mask=True,
        description="Vision-as-LoRA with bidirectional attention for vision tokens"
    )
    config.training.target_samples = 400_000
    config.training.run_name = "vision-lora-bidirectional"
    return config


def get_ablation_causal_config() -> Config:
    """Configuration for ablation study with causal masking."""
    config = Config()
    config.experiment = ExperimentConfig(
        name="vision_lora_causal",
        use_vision=True,
        use_bidirectional_mask=False,
        description="Vision-as-LoRA with causal attention mask (ablation)"
    )
    config.training.target_samples = 100_000
    config.training.run_name = "vision-lora-causal"
    return config