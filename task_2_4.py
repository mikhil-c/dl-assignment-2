import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from models.classification import VGG11Classifier
from data.pets_dataset import OxfordIIITPetDataset

# 1. Initialize W&B
wandb.init(project="da6401-assignment-2", name="task-2.4")

# 2. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG11Classifier(num_classes=37, in_channels=3).to(device)

# Load the classifier checkpoint
ckpt_path = "checkpoints/classifier.pth"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
model.eval()

# 3. Setup PyTorch Forward Hooks to extract feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# In your VGG11 features sequential:
# Index 0 is the 1st Conv2d layer (First Conv)
model.features[0].register_forward_hook(get_activation('conv_first'))
# Index 27 is the last Conv2d before the final MaxPool2d (Last Conv)
model.features[27].register_forward_hook(get_activation('conv_last'))

# 4. Search for a specific DOG image (Basset Hound for big ears/snout)
dataset_root = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
dataset = OxfordIIITPetDataset(root_dir=dataset_root, split='train')

target_dog_breed = 'basset_hound'
dog_idx = 0

for i, file_name in enumerate(dataset.image_files):
    if file_name.startswith(target_dog_breed):
        dog_idx = i
        break

# Grab the dog image (the dataset will apply your A.Normalize and resize to 224x224)
img_tensor, label_tensor, _, _ = dataset[dog_idx]
img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension: [1, 3, 224, 224]

print(f"Passing {dataset.image_files[dog_idx]} (Breed: {dataset.classes[label_tensor.item()]}) through the model...")

# 5. Perform the Forward Pass
with torch.no_grad():
    _ = model(img_tensor)

# 6. Visualization Helper Function
def plot_feature_maps(feature_map, title, num_cols=8, max_plots=64):
    """Plots the channels of a feature map in a grid."""
    # Move to CPU and remove batch dimension -> [C, H, W]
    fm = feature_map.squeeze(0).cpu()
    num_channels = fm.shape[0]
    
    # We'll plot up to 'max_plots' channels to keep the figure readable
    num_plots = min(num_channels, max_plots)
    num_rows = int(np.ceil(num_plots / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_plots:
            # Using viridis colormap to clearly show activation intensity
            ax.imshow(fm[i].numpy(), cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    return fig

# Generate the plots
fig_first = plot_feature_maps(activation['conv_first'], f'First Conv Layer - {target_dog_breed.replace("_", " ").title()}')
# The last layer has 512 channels, we plot the first 64 for visualization
fig_last = plot_feature_maps(activation['conv_last'], f'Last Conv Layer - {target_dog_breed.replace("_", " ").title()}')

# 7. Log to W&B
wandb.log({
    "First Conv Layer Maps": wandb.Image(fig_first),
    "Last Conv Layer Maps": wandb.Image(fig_last)
})

print("Successfully generated feature maps and logged to W&B!")
wandb.finish()

# Display in notebook as well
plt.show()
