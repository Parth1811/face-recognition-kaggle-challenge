import os

labels = []
with open("train.csv", "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        idx, img_path, label = line.strip().split(",")
        labels.append((img_path, label))

categories = []
with open("category.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        idx, category = line.strip().split(",")
        categories.append(category)


for category in categories:
    os.makedirs(f"train/{category}", exist_ok=True)

for img_path, label in labels:
    os.rename(f"train/{img_path}", f"train/{label}/{img_path}")



    
    