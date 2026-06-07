import torch
import torchvision
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"GPU dostępne: {torch.cuda.is_available()}")
