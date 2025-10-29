#!/bin/bash

echo "=================================="
echo "Setting up Vision-as-LoRA on Colab"
echo "=================================="

echo -e "\n1. Installing dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers peft datasets accelerate wandb pillow tqdm bitsandbytes

echo -e "\n2. Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo -e "\n3. Cloning repository..."
if [ -d "smol-vision-lora" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd smol-vision-lora
    git pull
else
    git clone https://github.com/arantesvictorl/smol-vision-lora.git
    cd smol-vision-lora
fi

echo -e "\n4. Setup complete!"
echo "Run: python scripts/colab_quick_test.py"