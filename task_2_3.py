import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.segmentation import VGG11UNet

def calculate_dice(pred, target, num_classes=3):
    pred_labels = torch.argmax(pred, dim=1)
    dice_scores = []
    for cls in range(num_classes):
        pred_mask = (pred_labels == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum().float() + target_mask.sum().float()
        if union > 0:
            dice_scores.append((2. * intersection) / union)
    return sum(dice_scores).item() / len(dice_scores) if len(dice_scores) > 0 else 0.0

def train_ablation(strategy, device, train_loader, val_loader, classifier_path):
    wandb.init(project="da6401-assignment-2", name=f"Task-2.3-{strategy}", job_type="train")
    
    model = VGG11UNet(num_classes=3, in_channels=3).to(device)
    
    # Load Pretrained Classifier Weights
    if os.path.exists(classifier_path):
        temp_classifier = VGG11Classifier(num_classes=37, in_channels=3).to(device)
        ckpt = torch.load(classifier_path, map_location=device)
        temp_classifier.load_state_dict(ckpt.get("state_dict", ckpt))

        classifier_params = list(temp_classifier.features.parameters())
        unet_encoders = [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]
        unet_params = []
        for enc in unet_encoders:
            unet_params.extend(list(enc.parameters()))

        for c_param, u_param in zip(classifier_params, unet_params):
            u_param.data.copy_(c_param.data)
            
            # APPLY THE STRATEGY
            if strategy == "Strict_Frozen":
                u_param.requires_grad = False
            elif strategy == "Full_Finetune":
                u_param.requires_grad = True

        if strategy == "Partial_Finetune":
            # Freeze enc1, enc2, enc3
            for enc in [model.enc1, model.enc2, model.enc3]:
                for param in enc.parameters():
                    param.requires_grad = False
            # Unfreeze enc4, enc5
            for enc in [model.enc4, model.enc5]:
                for param in enc.parameters():
                    param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    epochs = 15 

    print(f"\n--- Starting Training: {strategy} ---")
    for epoch in range(epochs):
        model.train()
        train_loss, train_dice = 0.0, 0.0
        
        for images, _, _, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += calculate_dice(outputs, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, _, _, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_dice += calculate_dice(outputs, masks)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch {epoch+1} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f}")
        wandb.log({
            "Train Loss": avg_train_loss, "Train Dice": avg_train_dice,
            "Val Loss": avg_val_loss, "Val Dice": avg_val_dice
        })

    wandb.finish()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_dir = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
    ckpt_path = "./checkpoints/classifier.pth" 
    
    train_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='train')
    val_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='test') 
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    strategies = ["Strict_Frozen", "Partial_Finetune", "Full_Finetune"]
    for strat in strategies:
        train_ablation(strat, device, train_loader, val_loader, ckpt_path)
