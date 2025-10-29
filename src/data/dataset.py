import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, Any
import random

from configs.config import Config


class VisionLanguageDataset(Dataset):
    """
    Dataset for vision-language training using COCO captions.
    
    Loads and preprocesses image-caption pairs for training multimodal models.
    """
    
    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        self.split = split
        
        split_config = (
            config.data.train_split if split == "train" 
            else config.data.val_split
        )
        
        self.dataset = load_dataset(
            config.data.dataset_name,
            split=split_config,
        )
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.transform = self._create_transform()
        
    def _create_transform(self):
        """Create image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(
                self.config.vision.image_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(self.config.vision.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing pixel_values, input_ids, and labels
        """
        try:
            item = self.dataset[idx]
            
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            pixel_values = self.transform(image)
            
            caption = self._extract_caption(item)
            
            text = self._format_text(caption)
            
            tokens = self.tokenizer(
                text,
                max_length=self.config.training.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = tokens['input_ids'].squeeze(0)
            labels = self._create_labels(input_ids, text)
            
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def _extract_caption(self, item: Dict[str, Any]) -> str:
        """Extract caption from dataset item."""
        sentences = item['sentences']
        if isinstance(sentences, list):
            caption = sentences[0]['raw']
        else:
            caption = sentences['raw'][0]
        return caption
    
    def _format_text(self, caption: str) -> str:
        """Format text for training."""
        return f"Describe: {caption}"
    
    def _create_labels(self, input_ids: torch.Tensor, text: str) -> torch.Tensor:
        """
        Create labels with proper masking.
        
        Masks the prompt portion so loss is only computed on the caption.
        """
        labels = input_ids.clone()
        
        prompt = "Describe: "
        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=False
        )['input_ids']
        
        if len(prompt_tokens) > 0:
            labels[:len(prompt_tokens)] = -100
            
        return labels


class TextOnlyDataset(Dataset):
    """Dataset for text-only baseline experiments."""
    
    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        self.split = split
        
        split_config = (
            config.data.train_split if split == "train" 
            else config.data.val_split
        )
        
        self.dataset = load_dataset(
            config.data.dataset_name,
            split=split_config,
        )
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get text-only item."""
        try:
            item = self.dataset[idx]
            
            sentences = item['sentences']
            if isinstance(sentences, list):
                caption = sentences[0]['raw']
            else:
                caption = sentences['raw'][0]
            
            text = f"Describe: {caption}"
            
            tokens = self.tokenizer(
                text,
                max_length=self.config.training.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = tokens['input_ids'].squeeze(0)
            labels = input_ids.clone()
            
            prompt = "Describe: "
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
            if len(prompt_tokens) > 0:
                labels[:len(prompt_tokens)] = -100
            
            return {
                'input_ids': input_ids,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))