import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

from configs.config import Config
from src.model.vision_lora_model import VisionLoRAModel


class Evaluator:
    """
    Evaluation utilities for vision-language models.
    
    Computes various metrics including perplexity and generation quality.
    """
    
    def __init__(self, model: VisionLoRAModel, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def compute_perplexity(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 100
    ) -> float:
        """
        Compute perplexity on dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Computing perplexity")):
                if i >= max_batches:
                    break
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                
                loss = outputs.loss
                
                num_tokens = (batch['labels'] != -100).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def generate_captions(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 10,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Generate captions for images.
        
        Args:
            dataloader: DataLoader with image-caption pairs
            num_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            List of (generated, reference) caption pairs
        """
        results = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                pixel_values = batch['pixel_values'].to(self.device)
                
                prompt = "Describe:"
                prompt_ids = self.model.tokenizer(
                    prompt,
                    return_tensors="pt"
                )['input_ids'].to(self.device)
                
                if self.config.experiment.use_vision:
                    vision_embeds = self.model.vision_embed(pixel_values)
                    prompt_embeds = self.model.llm.get_input_embeddings()(prompt_ids)
                    inputs_embeds = torch.cat([vision_embeds, prompt_embeds], dim=1)
                    
                    outputs = self.model.llm.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                    )
                else:
                    outputs = self.model.llm.generate(
                        input_ids=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                    )
                
                generated = self.model.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                reference_ids = batch['labels'][0]
                reference_tokens = reference_ids[reference_ids != -100]
                reference = self.model.tokenizer.decode(
                    reference_tokens,
                    skip_special_tokens=True
                )
                
                results.append((generated, reference))
        
        return results
    
    def evaluate_full(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 100
    ) -> Dict[str, float]:
        """
        Run complete evaluation suite.
        
        Args:
            dataloader: DataLoader for evaluation
            max_batches: Maximum batches for perplexity computation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        perplexity = self.compute_perplexity(dataloader, max_batches)
        metrics['perplexity'] = perplexity
        
        caption_pairs = self.generate_captions(dataloader, num_samples=10)
        
        print("\n" + "="*60)
        print("Sample Generations:")
        print("="*60)
        for i, (generated, reference) in enumerate(caption_pairs[:5], 1):
            print(f"\nSample {i}:")
            print(f"Generated: {generated}")
            print(f"Reference: {reference}")
            print("-"*60)
        
        return metrics


def evaluate_model(
    model: VisionLoRAModel,
    config: Config,
    dataloader: torch.utils.data.DataLoader,
) -> Dict[str, float]:
    """
    Convenience function for model evaluation.
    
    Args:
        model: Model to evaluate
        config: Configuration
        dataloader: Data for evaluation
        
    Returns:
        Evaluation metrics
    """
    evaluator = Evaluator(model, config)
    metrics = evaluator.evaluate_full(dataloader)
    return metrics