import torch
from torchvision import datasets, transforms
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from custom_dataset import CustomDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=224, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', classify=True, device=device)

num_classes = 100
fc_layer = nn.Linear(512, num_classes)
resnet.logits = fc_layer
resnet.load_state_dict(torch.load('model_2.pth'))

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


dataset = datasets.ImageFolder('train', transform=transform)
dataset = CustomDataset('processed_train')
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=150, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"Epoch: {epoch}, Batch: {i+1}/{len(train_loader)}, Train Loss: {loss.item()}")
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()
        

            loss = loss_fn(outputs, y)
            val_loss += loss.item()
        val_loss /= len(val_loader)

        accuracy = 100 * correct / total
        print(f'Accuracy on validation set: {accuracy:.2f}%')
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        torch.save(model.state_dict(), f"model_{epoch}.pth")

if __name__ == "__main__":
    model = resnet.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs=10)   