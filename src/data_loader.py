import torch
from torch.utils.data import Dataset
from PIL import Image
import os, pickle

class VQADataset(Dataset):
    def __init__(self, data_path, image_dir, word2idx, ans2idx, transform=None, max_len=20):
        self.data = pickle.load(open(data_path, "rb"))
        self.image_dir = image_dir
        self.word2idx = word2idx
        self.ans2idx = ans2idx
        self.transform = transform
        self.max_len = max_len

    def encode_question(self, question):
        words = question.split()
        ids = [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]
        if len(ids) < self.max_len:
            ids += [self.word2idx['<pad>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def encode_answer(self, answers):
        counts = {}
        for a in answers:
            idx = self.ans2idx.get(a, self.ans2idx['<unk>'])
            counts[idx] = counts.get(idx, 0) + 1
        return torch.tensor(max(counts, key=counts.get))

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.image_dir, f"COCO_train2014_{item['image_id']:012d}.jpg")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        q = self.encode_question(item["question"])
        a = self.encode_answer(item["answers"])
        return img, q, a

    def __len__(self):
        return len(self.data)
