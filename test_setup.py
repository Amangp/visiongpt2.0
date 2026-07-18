"""
Simple test script to verify the VQA model and inference setup
Run this to test if everything is working before starting the frontend
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def main():
    print("="*60)
    print("VQA Model Setup Verification")
    print("="*60)
    print()
    
    all_good = True

    model_path = os.environ.get("VQA_MODEL_PATH", "vqa_transformer.pth")
    word2idx_path = os.environ.get("VQA_WORD2IDX_PATH", "word2idx.pkl")
    ans2idx_path = os.environ.get("VQA_ANS2IDX_PATH", "ans2idx.pkl")
    
    # Check model file
    print("📦 Checking Model Files:")
    all_good &= check_file(model_path, "Model weights")
    print()
    
    # Check vocabulary files
    print("📚 Checking Vocabulary Files:")
    all_good &= check_file(word2idx_path, "Word vocabulary")
    all_good &= check_file(ans2idx_path, "Answer vocabulary")
    print()
    
    # Check frontend files
    print("🌐 Checking App Files:")
    all_good &= check_file("app.py", "Gradio app")
    all_good &= check_file("inference.py", "Inference module")
    print()
    
    # Check Python packages
    print("📦 Checking Python Packages:")
    try:
        import torch
        print(f"✅ PyTorch installed (version {torch.__version__})")
    except ImportError:
        print("❌ PyTorch NOT installed")
        all_good = False
    
    try:
        import torchvision
        print(f"✅ torchvision installed (version {torchvision.__version__})")
    except ImportError:
        print("❌ torchvision NOT installed")
        all_good = False
    
    try:
        import gradio
        print(f"✅ Gradio installed (version {gradio.__version__})")
    except ImportError:
        print("❌ Gradio NOT installed (run: pip install gradio)")
        all_good = False
    
    try:
        from PIL import Image
        print("✅ Pillow installed")
    except ImportError:
        print("❌ Pillow NOT installed")
        all_good = False
    
    # Try loading the model
    if all_good:
        print("🧪 Testing Model Loading:")
        try:
            from inference import VQAInference
            
            vqa = VQAInference(
                model_path=model_path,
                word2idx_path=word2idx_path,
                ans2idx_path=ans2idx_path,
            )
            print("✅ Model loaded successfully!")
            print(f"   Device: {vqa.device}")
            print(f"   Vocabulary size: {len(vqa.word2idx)}")
            print(f"   Answer vocabulary size: {len(vqa.ans2idx)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            all_good = False
    
    print()
    print("="*60)
    
    if all_good:
        print("🎉 SUCCESS! Everything is ready!")
        print()
        print("To start the frontend, run:")
        print("  python app.py")
        print()
    else:
        print("⚠️  Some issues found. Please fix them before starting the frontend.")
        print()
        print("Common fixes:")
        print("  - Make sure vqa_transformer.pth is in the project root")
        print("  - Make sure word2idx.pkl and ans2idx.pkl are in the project root")
        print("  - Run: pip install -r requirements.txt")
        print()
    
    print("="*60)

if __name__ == "__main__":
    main()
