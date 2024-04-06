import torch
from torchvision import datasets, transforms
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=224, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', classify=True, device=device)

num_classes = 100
fc_layer = nn.Linear(512, num_classes)
resnet.logits = fc_layer
resnet.load_state_dict(torch.load('model_0.pth'))  

class MTCNNTransform:
    def __init__(self, mtcnn, save_img=False):
        self.mtcnn = mtcnn
        self.save_img = save_img

    def __call__(self, img):
        # print(img, dir(img), type(img))
        
        # if self.save_img:
        #     return self.mtcnn(img)
        cropped = self.mtcnn(img)
        if cropped is None:
            img = img.resize((224, 224))
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        return cropped


transform = transforms.Compose([
    MTCNNTransform(mtcnn),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('train_small', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

correct = 0
total = 0

with torch.no_grad():
    for idx, (images, labels) in enumerate(val_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        print(f'Completed iteration {idx+1}/{len(val_dataloader)}')



accuracy = 100 * correct / total
print(f'Accuracy on validation set: {accuracy:.2f}%')