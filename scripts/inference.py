import argparse
import torch
from PIL import Image
from torchvision import transforms

from configs.config import get_vision_lora_config
from src.model.vision_lora_model import VisionLoRAModel


def load_image(image_path: str, image_size: int = 224):
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    pixel_values = transform(image).unsqueeze(0)
    
    return pixel_values


def main():
    parser = argparse.ArgumentParser(description="Run inference on image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    config = get_vision_lora_config()
    
    print("Loading model")
    model = VisionLoRAModel(config)
    model.eval()
    model.cuda()
    
    print(f"Loading image: {args.image}")
    pixel_values = load_image(args.image, config.vision.image_size)
    pixel_values = pixel_values.cuda()
    
    prompt = "Describe:"
    prompt_ids = model.tokenizer(
        prompt,
        return_tensors="pt"
    )['input_ids'].cuda()
    
    print("Generating caption")
    with torch.no_grad():
        vision_embeds = model.vision_embed(pixel_values)
        prompt_embeds = model.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([vision_embeds, prompt_embeds], dim=1)
        
        outputs = model.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    
    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("Generated Caption:")
    print("="*60)
    print(generated)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()