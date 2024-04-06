
import torch
from torchvision import datasets, transforms
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from PIL import Image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=224, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', classify=True, device=device)

num_classes = 100
fc_layer = nn.Linear(512, num_classes)
resnet.logits = fc_layer
resnet.load_state_dict(torch.load('model_0.pth'))  


images = os.listdir("test/test/")
images.sort(key= lambda x: int(x.split('.jpg')[0]))

dataset = datasets.ImageFolder('train_small')

resnet.eval()
results = []
with torch.no_grad():
    for idx, img_path in enumerate(images):
        with Image.open(f"test/test/{img_path}") as img:
            error  = False
            try:
                img_cropped = mtcnn(img)
            except:
                # print(f"Error in image: {img_path}")
                img = img.convert('RGB')
                img_cropped = mtcnn(img)

            if img_cropped is None:
                img = img.resize((224, 224))
                img = transforms.ToTensor()(img)
                img_cropped = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            img_cropped = img_cropped.to(device)
            img_cropped = img_cropped.unsqueeze(0)
            outputs = resnet(img_cropped)
            _, predicted = torch.max(outputs.data, 1)
            results.append((img_path, dataset.classes[predicted.item()]))
            print(f"Iteration: {idx+1}/{len(images)}, Predicted Class: {dataset.classes[predicted.item()]}")


print(results)
with open("test_out_full_epooch_6.csv", "w") as out_file:
    out_file.write("ID,Category\n")
    for path, cl in results:
        out_file.write(f"{path.split('.jpg')[0]},{cl}\n")
        