import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# 1. Define Your Model
# ------------------
# Must match the architecture used when saving the model
class SSM(torch.nn.Module):
    def __init__(self, num_classes=21):
        super(SSM, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Ensure final size is 256x256
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        return x

# ------------------
# 2. Load the Saved Model
# ------------------
def load_model(checkpoint_path='model_final.pth', num_classes=21, device='cpu'):
    model = SSM(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ------------------
# 3. Create a Color Palette (21 classes + background)
# ------------------
# You can adjust these to match official VOC colors or pick your own.
# This is just an example palette for 21 classes (0..20).
VOC_COLORMAP = [
    (0, 0, 0),        # class 0 (background)
    (128, 0, 0),      # class 1
    (0, 128, 0),      # class 2
    (128, 128, 0),    # class 3
    (0, 0, 128),      # class 4
    (128, 0, 128),    # class 5
    (0, 128, 128),    # class 6
    (128, 128, 128),  # class 7
    (64, 0, 0),       # class 8
    (192, 0, 0),      # class 9
    (64, 128, 0),     # class 10
    (192, 128, 0),    # class 11
    (64, 0, 128),     # class 12
    (192, 0, 128),    # class 13
    (64, 128, 128),   # class 14
    (192, 128, 128),  # class 15
    (0, 64, 0),       # class 16
    (128, 64, 0),     # class 17
    (0, 192, 0),      # class 18
    (128, 192, 0),    # class 19
    (0, 64, 128)      # class 20
]

def colorize_mask(mask):
    """
    Converts a 2D array of class indices into a color RGB image.
    mask: (H, W) with integer values in [0..20].
    """
    # Create an empty RGB image
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx in range(len(VOC_COLORMAP)):
        color = VOC_COLORMAP[class_idx]
        # Where the mask is the class index, fill in the color
        color_image[mask == class_idx] = color

    return color_image

# ------------------
# 4. Load the Dataset
# ------------------
# Must use the same transforms as during training
def get_voc_dataset(split='val'):
    common_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set=split,
        download=False,
        transform=common_transform,
        target_transform=common_transform
    )
    return dataset

# ------------------
# 5. Demo: Display a Few Samples
# ------------------
def display_predictions(model, dataset, device, num_samples=3):
    """
    Pick a few random samples from the dataset,
    run the model, and display original image, ground truth, and predicted mask.
    """
    import random
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for row_idx, idx in enumerate(indices):
        # Load a single sample
        image, target = dataset[idx]  # target is the ground truth mask
        
        # Prepare for inference
        image_batch = image.unsqueeze(0).to(device)  # (1, 3, 256, 256)
        
        with torch.no_grad():
            # Forward pass
            logits = model(image_batch)               # shape => (1, num_classes, 256, 256)
            pred = torch.argmax(logits, dim=1)        # (1, 256, 256)
            pred = pred.squeeze(0).cpu().numpy()      # (256, 256)
        
        # Convert ground truth to numpy & remove extra dim
        gt_mask = target.squeeze().numpy()
        
        # Convert to [H, W] of integer class indices
        # (If there's a channel dimension = 1, we remove it with .squeeze())
        
        # Colorize ground truth (ignoring label 255 => void if present)
        # For demonstration, let's map void to background color 0
        gt_mask[gt_mask == 255] = 0
        gt_color = colorize_mask(gt_mask.astype(int))
        
        # Colorize predicted mask
        pred_color = colorize_mask(pred.astype(int))

        # Show original image
        axes[row_idx, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[row_idx, 0].set_title("Original Image")
        axes[row_idx, 0].axis("off")
        
        # Show ground truth
        axes[row_idx, 1].imshow(gt_color)
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 1].axis("off")
        
        # Show prediction
        axes[row_idx, 2].imshow(pred_color)
        axes[row_idx, 2].set_title("Prediction")
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------
# 6. Main Entry
# ------------------
if __name__ == "__main__":
    # Choose CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load the model
    model = load_model(checkpoint_path='model_final.pth', num_classes=21, device=device)

    # 2) Load the PascalVOC val dataset
    val_dataset = get_voc_dataset(split='val')

    # 3) Display a few predictions
    display_predictions(model, val_dataset, device, num_samples=3)
