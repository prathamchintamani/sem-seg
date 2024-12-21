import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

class ssm(nn.Module):
    def __init__(self, num_classes):
        super(ssm, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            F.relu(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            F.relu(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=2,stride=2),
            F.relu(),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()

train_dataset = datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='train',
    download=True,
    transform=transform,
    target_transform=target_transform
)

val_dataset = datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='val',
    download=True,
    transform=transform,
    target_transform=target_transform
)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

