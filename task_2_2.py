import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier

def train_dropout_ablation(p_value, device, train_loader, val_loader, epochs=15):
    wandb.init(project="da6401-assignment-2", name=f"Task-2.2-Dropout-p{p_value}", job_type="train")
    
    # Initialize model with specific dropout probability
    model = VGG11Classifier(num_classes=37, in_channels=3, dropout_p=p_value).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"\n--- Training with Dropout p={p_value} ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        wandb.log({
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "Val Accuracy": val_acc
        })

    wandb.finish()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
    
    train_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='train')
    val_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    for p in [0.0, 0.2, 0.5]:
        train_dropout_ablation(p, device, train_loader, val_loader)
