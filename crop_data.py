import os
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

mtcnn = MTCNN(image_size=224, margin=0)

os.makedirs("processed_train", exist_ok=True)
for folder in tqdm(os.listdir("train")):
    os.makedirs(f"processed_train/{folder}", exist_ok=True)
    for img_path in tqdm(os.listdir(f"train/{folder}")):
        with Image.open(f"train/{folder}/{img_path}") as img:
            try:
                img_cropped = mtcnn(img)
            except RuntimeError:
                img = img.convert('RGB')
                img_cropped = mtcnn(img)
            if img_cropped is None:
                img = img.resize((224, 224))
                img = img.resize((224, 224))
                img = transforms.ToTensor()(img)
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                torch.save(img, f"processed_train/{folder}/{img_path.split('.')[0]}.pt")
            else:
                torch.save(img_cropped, f"processed_train/{folder}/{img_path.split('.')[0]}.pt")
