import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb
import os
from typing import Optional

from configs.config import Config
from src.model.vision_lora_model import VisionLoRAModel
from src.data.dataset import VisionLanguageDataset, TextOnlyDataset


class Trainer:
    """
    Training orchestrator for Vision-as-LoRA experiments.
    
    Handles the complete training loop including optimization, logging,
    evaluation, and checkpointing.
    """
    
    def __init__(
        self,
        model: VisionLoRAModel,
        config: Config,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self._setup_dataloaders()
        self._setup_optimization()
        self._setup_logging()
        
        self.global_step = 0
        self.epoch = 0
        
        if config.training.use_torch_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def _setup_dataloaders(self):
        """Initialize data loaders."""
        if self.train_dataset is not None:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                persistent_workers=self.config.data.persistent_workers,
                prefetch_factor=self.config.data.prefetch_factor,
            )
        
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.training.batch_size,
                num_workers=min(4, self.config.data.num_workers),
                pin_memory=self.config.data.pin_memory,
            )
    
    def _setup_optimization(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.config.training.weight_decay,
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.max_steps,
        )
        
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.training.use_bf16
        )
    
    def _setup_logging(self):
        """Initialize logging infrastructure."""
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.training.run_name,
            config=self.config.to_dict(),
        )
    
    def train(self):
        """Execute complete training loop."""
        print(f"Starting training: {self.config.experiment.name}")
        print(f"Max steps: {self.config.training.max_steps}")
        print(f"Effective batch size: {self.config.training.effective_batch_size}")
        
        self.model.print_trainable_parameters()
        
        self.model.train()
        
        progress_bar = tqdm(
            total=self.config.training.max_steps,
            desc="Training"
        )
        
        accumulation_step = 0
        
        while self.global_step < self.config.training.max_steps:
            for batch in self.train_loader:
                loss = self._training_step(batch, accumulation_step)
                
                accumulation_step += 1
                
                if accumulation_step % self.config.training.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    accumulation_step = 0
                    
                    if self.global_step % self.config.training.logging_steps == 0:
                        self._log_metrics(loss)
                    
                    if self.global_step % self.config.training.eval_steps == 0:
                        self._evaluate()
                    
                    if self._should_save_checkpoint():
                        self._save_checkpoint()
                    
                    if self.global_step >= self.config.training.max_steps:
                        break
            
            self.epoch += 1
            
            if self.global_step >= self.config.training.max_steps:
                break
        
        progress_bar.close()
        
        self._save_checkpoint(final=True)
        
        print("Training complete")
    
    def _training_step(self, batch: dict, accumulation_step: int) -> float:
        """Execute single training step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with torch.cuda.amp.autocast(enabled=self.config.training.use_bf16):
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.training.gradient_accumulation_steps
        
        self.scaler.scale(loss).backward()
        
        return loss.item() * self.config.training.gradient_accumulation_steps
    
    def _optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    def _log_metrics(self, loss: float):
        """Log metrics to wandb."""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "train/step": self.global_step,
            "train/epoch": self.epoch,
        }
        
        wandb.log(metrics, step=self.global_step)
    
    def _evaluate(self):
        """Run evaluation on validation set."""
        if self.val_dataset is None:
            return
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = 50
        
        with torch.no_grad():
            for batch in self.val_loader:
                if num_batches >= max_eval_batches:
                    break
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.config.training.use_bf16):
                    outputs = self.model(**batch)
                    total_loss += outputs.loss.item()
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        wandb.log({
            "val/loss": avg_loss,
            "val/step": self.global_step,
        }, step=self.global_step)
        
        print(f"\nValidation loss: {avg_loss:.4f}\n")
        
        self.model.train()
    
    def _should_save_checkpoint(self) -> bool:
        """Determine if checkpoint should be saved."""
        if self.config.training.save_total_limit <= 0:
            return False
        
        save_interval = self.config.training.max_steps // self.config.training.save_total_limit
        
        return self.global_step % save_interval == 0 and self.global_step > 0
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        if final:
            save_dir = os.path.join(
                self.config.training.output_dir,
                "final"
            )
        else:
            save_dir = os.path.join(
                self.config.training.output_dir,
                f"checkpoint-{self.global_step}"
            )
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        
        print(f"Saved checkpoint: {save_dir}")


def create_trainer(config: Config) -> Trainer:
    """
    Factory function to create configured trainer.
    
    Args:
        config: Complete configuration object
        
    Returns:
        Initialized trainer ready for training
    """
    model = VisionLoRAModel(config)
    
    if config.experiment.use_vision:
        train_dataset = VisionLanguageDataset(config, split="train")
        val_dataset = VisionLanguageDataset(config, split="val")
    else:
        train_dataset = TextOnlyDataset(config, split="train")
        val_dataset = TextOnlyDataset(config, split="val")
    
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    return trainer