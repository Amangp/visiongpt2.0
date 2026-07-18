"""
Script to pre-download ViT weights
"""
import torch
from torchvision import models
import gc

print("Downloading ViT-B/16 weights...")
print("This may take a few minutes...")

try:
    # Force garbage collection before loading
    gc.collect()
    
    # Load with minimal memory footprint
    with torch.no_grad():
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    print("✅ ViT weights downloaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in vit.parameters()):,}")
    
    # Clean up
    del vit
    gc.collect()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
