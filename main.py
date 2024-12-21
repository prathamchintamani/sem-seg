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
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=2,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 21, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True) 

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

train_dataset = datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='train',
    download=False,
    transform=transform,
    target_transform=target_transform
)

val_dataset = datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='val',
    download=False,
    transform=transform,
    target_transform=target_transform
)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()

model = ssm(num_classes=21)
num_epochs = 20

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets.squeeze())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

