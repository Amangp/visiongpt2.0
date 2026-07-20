import pickle

with open("data/processed/train_data.pkl", "rb") as f:
    data = pickle.load(f)

filename = "COCO_train2014_000000000009.jpg"   # replace with the filename from ls

image_id = int(filename.split("_")[-1].split(".")[0])

for sample in data:
    if sample["image_id"] == image_id:
        print(sample)
        break