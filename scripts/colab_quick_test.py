import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.colab_config import get_colab_test_config
from src.model.vision_lora_model import VisionLoRAModel
from src.data.dataset import VisionLanguageDataset
from torch.utils.data import DataLoader


def quick_test():
    """Run quick validation test."""
    
    print("="*60)
    print("COLAB A100 QUICK TEST")
    print("="*60)
    
    config = get_colab_test_config()
    
    print("\n1. Checking GPU...")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    print("\n2. Loading model...")
    model = VisionLoRAModel(config)
    model.print_trainable_parameters()
    
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Memory allocated: {memory_allocated:.2f} GB")
    
    print("\n3. Loading dataset...")
    dataset = VisionLanguageDataset(config, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    
    print("\n4. Testing forward pass...")
    batch = next(iter(loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(**batch)
    
    print(f"Loss: {outputs.loss.item():.4f}")
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory: {peak_memory:.2f} GB")
    
    print("\n5. Testing backward pass...")
    outputs.loss.backward()
    print("Backward pass successful")
    
    print("\n6. Performance estimate...")
    import time
    
    model.eval()
    times = []
    
    for _ in range(5):
        batch = next(iter(loader))
        batch = {k: v.cuda() for k, v in batch.items()}
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model(**batch)
        
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    samples_per_sec = 4 / avg_time
    
    print(f"Avg time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    
    batch_size = 24
    gradient_accum = 4
    effective_batch = batch_size * gradient_accum
    
    target_samples = 250_000
    steps = target_samples // effective_batch
    
    step_time = avg_time * gradient_accum * (batch_size / 4)
    total_hours = (steps * step_time) / 3600
    
    print(f"\nEstimated training time for {target_samples:,} samples:")
    print(f"Steps: {steps:,}")
    print(f"Time on A100: {total_hours:.1f} hours")
    print(f"Time on H100 (est): {total_hours/1.5:.1f} hours")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    quick_test()