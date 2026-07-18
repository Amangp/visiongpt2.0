# Visual Question Answering (VQA)

This repository is a Visual Question Answering demo: upload an image, ask a natural-language question, and the model returns the top answers with confidence scores.

It includes:
- `vqa_transformer.pth` (trained model checkpoint)
- `word2idx.pkl` and `ans2idx.pkl` (vocabulary mappings used at training time)
- A Gradio web app (`app.py`) for interactive inference

Note: the checkpoint file is large, so this repo uses Git LFS.

## What is VQA?

Visual Question Answering (VQA) is a multi-modal task that combines computer vision and NLP. The model takes:
- an image, and
- a text question (e.g. “How many people are there?”)

and predicts an answer from a fixed answer vocabulary (classification). In this project, the image is encoded with a Transformer-style vision backbone and the question is encoded with a text Transformer; their representations are fused and used to predict the most likely answer.

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Git LFS (required to download `vqa_transformer.pth`)

Install Git LFS:
- macOS (Homebrew): `brew install git-lfs`
- Windows: install from https://git-lfs.com/

After installing: `git lfs install`

## Setup (macOS)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python test_setup.py

python app.py
```

Open: http://127.0.0.1:7860

## Setup (Windows)

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python test_setup.py

python app.py
```

Or double-click: `start_frontend.bat`

Open: http://127.0.0.1:7860


## Optional: Override file locations

By default the app loads these files from the repo root:
- `vqa_transformer.pth`
- `word2idx.pkl`
- `ans2idx.pkl`

You can override paths with environment variables:
- `VQA_MODEL_PATH`
- `VQA_WORD2IDX_PATH`
- `VQA_ANS2IDX_PATH`

## 🔬 Model Performance

Expected performance on VQA v2.0 validation set:

- **Overall Accuracy**: 62.8%
- **Yes/No Questions**: ~78-82%
- **Number Questions**: ~42-46%
- **Other Questions**: ~55-58%

Performance can be improved with:
- Longer training (50+ epochs)
- Larger batch sizes (if GPU memory allows)
- Data augmentation
- Answer vocabulary expansion
- Ensemble methods
