import torch, pickle, yaml
from src.model import VQAModel

def evaluate():
    config = yaml.safe_load(open("config.yaml"))
    device = config['training']['device']

    with open("data/processed/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("data/processed/ans2idx.pkl", "rb") as f:
        ans2idx = pickle.load(f)

    model = VQAModel(
        len(word2idx), len(ans2idx),
        config['model']['embed_dim'],
        config['model']['hidden_dim'],
        config['model']['image_feat_dim'],
        config['model']['attention_dim']
    )
    model.load_state_dict(torch.load("vqa_model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded for evaluation")

if __name__ == "__main__":
    evaluate()
