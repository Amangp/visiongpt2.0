import json, re, os, pickle
from collections import Counter

def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

def load_data(questions_path, annotations_path):
    with open(questions_path) as fq, open(annotations_path) as fa:
        questions = json.load(fq)['questions']
        annotations = json.load(fa)['annotations']
    q_dict = {q['question_id']: q for q in questions}
    merged = []
    for ann in annotations:
        q = q_dict.get(ann['question_id'])
        if q:
            merged.append({
                "image_id": ann['image_id'],
                "question": normalize_text(q['question']),
                "answers": [normalize_text(a['answer']) for a in ann['answers']]
            })
    return merged

def build_vocab(data, min_word_count=3):
    counter = Counter()
    for item in data:
        counter.update(item['question'].split())
    vocab = [w for w, c in counter.items() if c >= min_word_count]
    word2idx = {w: i+1 for i, w in enumerate(vocab)}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = len(word2idx)
    return word2idx

def build_answer_vocab(data, top_n=3000):
    counter = Counter()
    for item in data:
        counter.update(item['answers'])
    most_common = counter.most_common(top_n)
    ans2idx = {a: i for i, (a, _) in enumerate(most_common)}
    ans2idx['<unk>'] = len(ans2idx)
    return ans2idx

if __name__ == "__main__":
    train_data = load_data(
        "data/questions/v2_OpenEnded_mscoco_train2014_questions.json",
        "data/annotations/v2_mscoco_train2014_annotations.json"
    )
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    word2idx = build_vocab(train_data)
    ans2idx = build_answer_vocab(train_data)
    pickle.dump(word2idx, open("data/processed/word2idx.pkl", "wb"))
    pickle.dump(ans2idx, open("data/processed/ans2idx.pkl", "wb"))
    print("✅ Preprocessing complete.")
