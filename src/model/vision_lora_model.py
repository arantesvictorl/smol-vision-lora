import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from configs.config import Config
from src.model.vision_embedding import VisionEmbedding


class VisionLoRAModel(nn.Module):
    """
    SmolLM2 with Vision-as-LoRA integration.
    
    Integrates vision capabilities into a small language model using LoRA layers
    applied to specific transformer blocks, allowing the model to process both
    visual and textual inputs.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self._init_llm()
        self._init_vision()
        self._apply_lora()
        
    def _init_llm(self):
        """Initialize base language model."""
        attn_implementation = "flash_attention_2" if self.config.model.use_flash_attention else "eager"
        
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_name,
                torch_dtype=getattr(torch, self.config.model.torch_dtype),
                attn_implementation=attn_implementation,
                device_map=self.config.model.device_map,
                trust_remote_code=self.config.model.trust_remote_code,
            )
        except ImportError as e:
            if "flash_attn" in str(e):
                print("Warning: Flash Attention not available, falling back to eager attention")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.config.model.model_name,
                    torch_dtype=getattr(torch, self.config.model.torch_dtype),
                    attn_implementation="eager",
                    device_map=self.config.model.device_map,
                    trust_remote_code=self.config.model.trust_remote_code,
                )
            else:
                raise
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.llm.parameters():
            param.requires_grad = False
            
    def _init_vision(self):
        """Initialize vision embedding layer."""
        if self.config.experiment.use_vision:
            self.vision_embed = VisionEmbedding(self.config.vision)
        else:
            self.vision_embed = None
            
    def _apply_lora(self):
        """Apply LoRA to specified layers."""
        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            layers_to_transform=list(range(self.config.lora.vision_layers)),
        )
        
        self.llm = get_peft_model(self.llm, lora_config)
        
    def create_attention_mask(
        self, 
        vision_len: int, 
        text_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Create attention mask based on experiment configuration.
        
        Args:
            vision_len: Number of vision tokens
            text_len: Number of text tokens
            device: Target device
            
        Returns:
            Attention mask tensor
        """
        total_len = vision_len + text_len
        
        if self.config.experiment.use_bidirectional_mask:
            mask = torch.ones(total_len, total_len, device=device, dtype=torch.bool)
            
            text_mask = torch.triu(
                torch.ones(text_len, text_len, device=device),
                diagonal=1
            ).bool()
            mask[vision_len:, vision_len:] = text_mask
            
            return ~mask
        else:
            mask = torch.triu(
                torch.ones(total_len, total_len, device=device),
                diagonal=1
            ).bool()
            return ~mask
    
    def forward(
        self,
        pixel_values: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Image tensor of shape (batch_size, channels, height, width)
            input_ids: Token IDs of shape (batch_size, sequence_length)
            labels: Target labels for language modeling loss
            
        Returns:
            Model outputs with loss and logits
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if self.vision_embed is not None and pixel_values is not None:
            vision_embeds = self.vision_embed(pixel_values)
            num_patches = vision_embeds.shape[1]
            
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            
            attention_mask = self.create_attention_mask(
                num_patches,
                input_ids.shape[1],
                device
            )
            attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            if labels is not None:
                vision_labels = torch.full(
                    (batch_size, num_patches),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )
                labels = torch.cat([vision_labels, labels], dim=1)
        else:
            inputs_embeds = text_embeds
            attention_mask = None
            
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        
        return outputs
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters."""
        self.llm.print_trainable_parameters()
        
        if self.vision_embed is not None:
            vision_params = sum(p.numel() for p in self.vision_embed.parameters())
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"Vision embedding parameters: {vision_params:,}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def save_pretrained(self, save_directory: str):
        """Save model weights."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.llm.save_pretrained(save_directory)
        
        if self.vision_embed is not None:
            vision_dir = os.path.join(save_directory, "vision")
            self.vision_embed.save_pretrained(vision_dir)