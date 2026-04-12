"""Training entrypoint
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
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

    wandb.init(project="da6401_assignment_2", name="vgg11_classification")

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
        wandb.log({
            "clf_loss": avg_loss, 
            "clf_macro_f1": f1,
            "clf_lr": optimizer.param_groups[0]['lr']
        })

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epochs
    }, "checkpoints/classifier.pth")
    wandb.finish()

def train_localizer(device: torch.device, train_loader: DataLoader, epochs: int = 15):
    model = VGG11Localizer(in_channels=3).to(device)
    
    classifier_path = "checkpoints/classifier.pth"
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    for param in model.features.parameters():
        param.requires_grad = False

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction='mean')
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    wandb.init(project="da6401_assignment_2", name="vgg11_localization")

    print("Starting Localization Training...")
    for epoch in range(epochs):
        model.train()
        
        model.features.eval()
        
        total_loss = 0.0

        for images, _, bboxes, _ in train_loader:
            images = images.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss_mse = mse_loss(outputs, bboxes)
            loss_iou = iou_loss(outputs, bboxes)
            loss = (0.001 * loss_mse) + loss_iou
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loc Loss: {epoch_loss:.4f}")
        wandb.log({"loc_total_loss": epoch_loss})

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epochs
    }, "checkpoints/localizer.pth")
    wandb.finish()

def train_segmentation(device: torch.device, train_loader: DataLoader, epochs: int = 15):
    model = VGG11UNet(num_classes=3, in_channels=3).to(device)
    
    classifier_path = "checkpoints/classifier.pth"
    if os.path.exists(classifier_path):
        temp_classifier = VGG11Classifier(num_classes=37, in_channels=3).to(device)
        ckpt = torch.load(classifier_path, map_location=device)
        temp_classifier.load_state_dict(ckpt.get("state_dict", ckpt))

        c_modules = [m for m in temp_classifier.features.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
        
        unet_encoders = [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]
        u_modules = []
        for enc in unet_encoders:
            u_modules.extend([m for m in enc.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))])

        for c_m, u_m in zip(c_modules, u_modules):
            u_m.load_state_dict(c_m.state_dict())
            for p in u_m.parameters():
                p.requires_grad = False
            
        print("Successfully transferred classifier weights and BN stats to UNet encoder!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    wandb.init(project="da6401_assignment_2", name="unet_segmentation")

    print("Starting Segmentation Training...")
    for epoch in range(epochs):
        model.train()
        
        for enc in [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]:
            enc.eval()
            
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

        epoch_loss = total_loss / len(train_loader)
        epoch_dice_avg = epoch_dice / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Seg Loss: {epoch_loss:.4f}, Dice: {epoch_dice_avg:.4f}")
        wandb.log({
            "seg_loss": epoch_loss,
            "seg_dice": epoch_dice_avg
        })

    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epochs
    }, "checkpoints/unet.pth")
    wandb.finish()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # Kaggle dataset path
    dataset_root = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
    dataset = OxfordIIITPetDataset(root_dir=dataset_root, split='train')
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    train_classifier(device, train_loader, epochs=50)
    gc.collect()
    torch.cuda.empty_cache()

    train_localizer(device, train_loader, epochs=30)
    gc.collect()
    torch.cuda.empty_cache()

    train_segmentation(device, train_loader, epochs=50)
