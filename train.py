"""Training entrypoint
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import gc

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> float:
    pred_labels = torch.argmax(pred, dim=1)
    dice_scores = []
    
    for cls in range(num_classes):
        pred_mask = (pred_labels == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum().float() + target_mask.sum().float()
        
        if union > 0:
            dice_scores.append((2. * intersection) / union)
            
    if len(dice_scores) == 0:
        return 0.0
    return sum(dice_scores).item() / len(dice_scores)

def train_classifier(device: torch.device, train_loader: DataLoader, epochs: int = 15):
    model = VGG11Classifier(num_classes=37, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    print("Starting Classification Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        f1 = f1_score(all_targets, all_preds, average='macro')
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, F1: {f1:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epochs
    }, "checkpoints/classifier.pth")

def train_localizer(device: torch.device, train_loader: DataLoader, epochs: int = 15):
    model = VGG11Localizer(in_channels=3).to(device)
    
    classifier_path = "checkpoints/classifier.pth"
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.features.load_state_dict(state_dict, strict=False)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Localization Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, _, bboxes, _ in train_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss_mse = mse_loss(outputs, bboxes)
            loss_iou = iou_loss(outputs, bboxes)
            loss = loss_mse + loss_iou
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loc Loss: {total_loss / len(train_loader):.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epochs
    }, "checkpoints/localizer.pth")

def train_segmentation(device: torch.device, train_loader: DataLoader, epochs: int = 15):
    model = VGG11UNet(num_classes=3, in_channels=3).to(device)
    
    classifier_path = "checkpoints/classifier.pth"
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Segmentation Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_dice = 0.0

        for images, _, _, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_dice += calculate_dice(outputs, masks)

        print(f"Epoch [{epoch+1}/{epochs}] - Seg Loss: {total_loss / len(train_loader):.4f}, Dice: {epoch_dice / len(train_loader):.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epochs
    }, "checkpoints/unet.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    dataset_root = "./dataset" 
    dataset = OxfordIIITPetDataset(root_dir=dataset_root, split='train')
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    train_classifier(device, train_loader, epochs=50)
    gc.collect()
    
    train_localizer(device, train_loader, epochs=30)
    gc.collect()
    
    train_segmentation(device, train_loader, epochs=50)
