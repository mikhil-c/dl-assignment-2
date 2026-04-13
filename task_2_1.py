import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier

def remove_batchnorm(model):
    """Replaces all BatchNorm layers with Identity to simulate training without BN."""
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(model, name, nn.Identity())
        else:
            remove_batchnorm(module)
    return model

def train_bn_ablation(use_bn, device, train_loader, val_loader, epochs=10):
    run_name = "Task-2.1-With-BatchNorm" if use_bn else "Task-2.1-No-BatchNorm"
    wandb.init(project="da6401-assignment-2", name=run_name, job_type="train")
    
    model = VGG11Classifier(num_classes=37, in_channels=3).to(device)
    if not use_bn:
        model = remove_batchnorm(model)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Setup forward hook for the 3rd Conv layer (features[8] in your architecture)
    activations = []
    def hook_fn(m, i, o):
        activations.append(o.detach().cpu())
    
    # Register hook
    hook_handle = model.features[8].register_forward_hook(hook_fn)

    print(f"\n--- Training {'WITH' if use_bn else 'WITHOUT'} BatchNorm ---")
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
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        wandb.log({"Train Loss": avg_train_loss})

    # After training, pass a single test image to capture final activation distribution
    model.eval()
    activations.clear() # Clear training activations
    with torch.no_grad():
        test_img, _, _, _ = next(iter(val_loader))
        model(test_img.to(device))
        
    # Log Histogram to W&B
    final_activations = activations[0].numpy()
    wandb.log({f"3rd Conv Layer Activations ({run_name})": wandb.Histogram(final_activations)})
    
    hook_handle.remove()
    wandb.finish()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
    
    train_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='train')
    val_dataset = OxfordIIITPetDataset(root_dir=root_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    train_bn_ablation(use_bn=False, device=device, train_loader=train_loader, val_loader=val_loader)
    train_bn_ablation(use_bn=True, device=device, train_loader=train_loader, val_loader=val_loader)
