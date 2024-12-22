import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------
# 1. Model Definition
# ------------------

class SSM(nn.Module):
    def __init__(self, num_classes=21):
        super(SSM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Note: We use 'num_classes' so it’s not hard-coded to 21.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Interpolate back to 256x256 if needed
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x


# ------------------------------
# 2. Data Transformations / Dataset
# ------------------------------
#
# For PASCAL VOC segmentation, label 255 denotes 'ignore'. 
# We must ensure that we apply the same spatial transforms (flip, rotation) 
# to both image and mask. If you do random transforms on images, you must 
# do the same on the masks. For simplicity, here we’ll just do:
#
#   - Random resize or just fixed resize (256,256)
#   - Convert to tensor
#
# Once you're comfortable, you can add synchronized random flips/rotations.

common_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# If you really want random flips/rotations, 
# you need a custom approach or a library that applies them identically 
# to both image & target. For debugging, let's keep them consistent as above.

train_dataset = datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='train',
    download=False,
    transform=common_transform,          # same transform for image
    target_transform=common_transform    # same transform for mask
)

val_dataset = datasets.VOCSegmentation(
    root='./data',
    year='2012',
    image_set='val',
    download=False,
    transform=common_transform,          # same transform for image
    target_transform=common_transform    # same transform for mask
)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------
# 3. Loss & Model
# ------------------

# Use ignore_index=255 for VOC
criterion = nn.CrossEntropyLoss(ignore_index=255)

model = SSM(num_classes=21)

# ------------------
# 4. Optimizer & Scheduler
# ------------------

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# We'll track the training loss or validation loss, 
# so the scheduler can reduce LR if loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

# ------------------
# 5. Training Setup
# ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Convert targets to Long tensor with shape [N, H, W]
        # typically VOCSegmentation loads masks as shape [N,1,H,W]
        # also ensure it's of type long
        if targets.ndim == 4:  
            targets = targets.squeeze(1)  # from [N,1,H,W] -> [N,H,W]
        targets = targets.long()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Average training loss this epoch
    train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")

    # Reduce LR on plateau (here we feed it the training loss)
    scheduler.step(train_loss)

    # -------------
    # Validation
    # -------------
    # You can evaluate on val_loader if you wish to monitor val loss:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if targets.ndim == 4:  
                targets = targets.squeeze(1)
            targets = targets.long()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"          Validation Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

torch.save(model.state_dict(), 'model_final.pth')
print("Model saved to 'model_final.pth'.")

