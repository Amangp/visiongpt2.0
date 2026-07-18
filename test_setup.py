"""
Simple test script to verify the TensorFlow VQA model and inference setup.

Run this before starting the Gradio frontend.
"""

import os


def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False


def main():

    print("=" * 60)
    print("TensorFlow VQA Setup Verification")
    print("=" * 60)
    print()

    all_good = True

    model_path = os.environ.get(
        "VQA_MODEL_PATH",
        "vqa_transformer.keras",
    )

    word2idx_path = os.environ.get(
        "VQA_WORD2IDX_PATH",
        "word2idx.pkl",
    )

    ans2idx_path = os.environ.get(
        "VQA_ANS2IDX_PATH",
        "ans2idx.pkl",
    )

    # --------------------------------------------------
    # Model Files
    # --------------------------------------------------

    print("📦 Checking Model Files:")

    all_good &= check_file(
        model_path,
        "TensorFlow model",
    )

    print()

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    print("📚 Checking Vocabulary Files:")

    all_good &= check_file(
        word2idx_path,
        "Word vocabulary",
    )

    all_good &= check_file(
        ans2idx_path,
        "Answer vocabulary",
    )

    print()

    # --------------------------------------------------
    # App Files
    # --------------------------------------------------

    print("🌐 Checking App Files:")

    all_good &= check_file(
        "app.py",
        "Gradio App",
    )

    all_good &= check_file(
        "inference.py",
        "Inference Module",
    )

    print()

    # --------------------------------------------------
    # Python Packages
    # --------------------------------------------------

    print("📦 Checking Python Packages:")

    try:
        import tensorflow as tf

        print(
            f"✅ TensorFlow installed (version {tf.__version__})"
        )

    except ImportError:

        print("❌ TensorFlow NOT installed")

        all_good = False

    try:
        import gradio

        print(
            f"✅ Gradio installed (version {gradio.__version__})"
        )

    except ImportError:

        print("❌ Gradio NOT installed")

        all_good = False

    try:
        from PIL import Image

        print("✅ Pillow installed")

    except ImportError:

        print("❌ Pillow NOT installed")

        all_good = False

    try:
        import numpy

        print(
            f"✅ NumPy installed (version {numpy.__version__})"
        )

    except ImportError:

        print("❌ NumPy NOT installed")

        all_good = False

    # --------------------------------------------------
    # Test Model Loading
    # --------------------------------------------------

    if all_good:

        print()
        print("🧪 Testing Model Loading:")

        try:

            from inference import VQAInference

            vqa = VQAInference(
                model_path=model_path,
                word2idx_path=word2idx_path,
                ans2idx_path=ans2idx_path,
            )

            print("✅ Model loaded successfully!")

            print(
                f"Vocabulary size: {len(vqa.word2idx)}"
            )

            print(
                f"Answer classes: {len(vqa.ans2idx)}"
            )

        except Exception as e:

            print(f"❌ Error loading model:\n{e}")

            all_good = False

    print()
    print("=" * 60)

    if all_good:

        print("🎉 SUCCESS! Everything is ready.")
        print()
        print("Run:")
        print("    python app.py")

    else:

        print("⚠️ Some issues were found.")
        print()
        print("Common fixes:")
        print("  • Ensure vqa_transformer.keras exists")
        print("  • Ensure word2idx.pkl exists")
        print("  • Ensure ans2idx.pkl exists")
        print("  • Install dependencies:")
        print("      pip install -r requirements.txt")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()