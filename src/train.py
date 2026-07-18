import torch, pickle, yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data_loader import VQADataset
from src.model import VQATransformer

def train():
    config = yaml.safe_load(open("config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥 Using device:", device)

    word2idx = pickle.load(open("data/processed/word2idx.pkl", "rb"))
    ans2idx  = pickle.load(open("data/processed/ans2idx.pkl", "rb"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = VQADataset("data/processed/train_data.pkl",
                         config['data']['image_dir'],
                         word2idx, ans2idx, transform)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                        shuffle=True, num_workers=4)

    model = VQATransformer(
        vocab_size=len(word2idx),
        ans_size=len(ans2idx),
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['training']['lr_decay'])

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, ques, ans in tqdm(loader, desc=f"Epoch {epoch+1}"):
            imgs, ques, ans = imgs.to(device), ques.to(device), ans.to(device)
            preds = model(imgs, ques)
            loss = criterion(preds, ans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred_idx = preds.max(1)
            correct += (pred_idx == ans).sum().item()
            total += ans.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%")
        scheduler.step()

    torch.save(model.state_dict(), "vqa_transformer.pth")
    print("✅ Saved model as vqa_transformer.pth")

if __name__ == "__main__":
    train()
