import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset

# 1. Initialize W&B
wandb.init(project="da6401-assignment-2", name="task-2.6")

# 2. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskPerceptionModel().to(device)
model.eval()

# 3. Setup Dataset
dataset_root = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
test_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split='test')
# shuffle=True to get a random set of 5 different samples
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 4. Inference and Table Logging
columns = ["Image_ID", "Original", "Ground_Truth", "Prediction", "Pixel_Accuracy", "Dice_Score"]
wb_table = wandb.Table(columns=columns)

count = 0
with torch.no_grad():
    for i, (img, label, bbox, mask) in enumerate(test_loader):
        if count >= 5: break
        
        img_cuda = img.to(device)
        outputs = model(img_cuda)
        seg_logits = outputs['segmentation']
        pred_mask = torch.argmax(seg_logits, dim=1).cpu().squeeze().numpy()
        gt_mask = mask.squeeze().numpy()

        # Metrics calculation
        acc = float((pred_mask == gt_mask).mean())
        
        dice_list = []
        for cls in range(3):
            p, t = (pred_mask == cls), (gt_mask == cls)
            if t.sum() + p.sum() > 0:
                dice_list.append(2.0 * (p & t).sum() / (p.sum() + t.sum()))
        dice = float(np.mean(dice_list)) if dice_list else 0.0

        # Image processing for display
        img_np = img.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # Scale classes (0, 1, 2) to visible grayscale values
        gt_display = (gt_mask * 127).astype(np.uint8)
        pred_display = (pred_mask * 127).astype(np.uint8)

        wb_table.add_data(
            f"test_sample_{i}",
            wandb.Image(img_np),
            wandb.Image(gt_display),
            wandb.Image(pred_display),
            round(acc, 4),
            round(dice, 4)
        )
        count += 1

wandb.log({"Segmentation_Samples_Table": wb_table})
print("Successfully logged 5 segmentation samples.")
wandb.finish()
