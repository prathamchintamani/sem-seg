import torch
from torchvision import models, transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
resize_dim = (256, 256)  # Resize all images and masks to a fixed size

transform_image = transforms.Compose([
    transforms.Resize(resize_dim),  # Resize to consistent dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize(resize_dim),  # Resize masks to match image dimensions
    transforms.ToTensor()
])

# Custom transform wrapper for dataset
class VOCSegmentationWithTransforms(VOCSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        image = transform_image(image)
        mask = transform_mask(mask)
        mask = (mask > 0).long()  # Convert mask to binary/long type if needed
        return image, mask

# Load VOCSegmentation dataset with transformations
train_dataset = VOCSegmentationWithTransforms(
    root='./data', year='2012', image_set='train', download=False
)
val_dataset = VOCSegmentationWithTransforms(
    root='./data', year='2012', image_set='val', download=False
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Pretrained segmentation model (FCN)
model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
model = model.to(device)
model.eval()  # For inference

def denormalize_image(image, mean, std):
    """Denormalize an image tensor for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    image = image * std + mean  # Reverse normalization
    return torch.clamp(image, 0, 1)  # Clip to valid range [0, 1]

def visualize_segmentation(image, mask, pred_mask):
    # Denormalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = denormalize_image(image, mean, std)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0).cpu())
    axes[0].set_title("Input Image")
    axes[1].imshow(mask.squeeze().cpu(), cmap='gray')
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred_mask.squeeze().cpu(), cmap='gray')
    axes[2].set_title("Predicted Mask")
    plt.show()


# Display some results
for i, (images, masks) in enumerate(val_loader):
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)['out']  # Forward pass

    # Convert predictions to masks
    predicted_masks = torch.argmax(outputs, dim=1)

    # Visualize the first batch
    for j in range(len(images)):
        visualize_segmentation(images[j], masks[j], predicted_masks[j])

    if i == 0:  # Visualize only the first batch
        break
