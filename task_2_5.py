import torch
import wandb
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordIIITPetDataset

# 1. Initialize W&B
wandb.init(project="da6401-assignment-2", name="task-2.5")

# 2. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskPerceptionModel().to(device)
model.eval()

# 3. Setup Dataset
dataset_root = "/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset"
test_dataset = OxfordIIITPetDataset(root_dir=dataset_root, split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 4. IoU Calculation Helper (Pixel Space)
def calculate_iou(pred, target):
    p_x1, p_y1 = pred[0] - pred[2]/2, pred[1] - pred[3]/2
    p_x2, p_y2 = pred[0] + pred[2]/2, pred[1] + pred[3]/2
    t_x1, t_y1 = target[0] - target[2]/2, target[1] - target[3]/2
    t_x2, t_y2 = target[0] + target[2]/2, target[1] + target[3]/2

    i_x1, i_y1 = max(p_x1, t_x1), max(p_y1, t_y1)
    i_x2, i_y2 = min(p_x2, t_x2), min(p_y2, t_y2)

    inter_w = max(0, i_x2 - i_x1)
    inter_h = max(0, i_y2 - i_y1)
    inter = inter_w * inter_h

    union = (pred[2]*pred[3]) + (target[2]*target[3]) - inter + 1e-6
    return float(inter / union)

# Convert [cx, cy, w, h] to [xmin, ymin, xmax, ymax] for PIL drawing
def get_corners(box):
    return [box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2]

# 5. Inference and Table Logging
columns = ["Image_ID", "Prediction_Overlay", "Confidence_Score", "IoU_Score"]
wb_table = wandb.Table(columns=columns)

count = 0
with torch.no_grad():
    for i, (img, label, target_bbox, _) in enumerate(test_loader):
        if count >= 10: break
        
        img_id = test_dataset.image_files[i]
        img_cuda = img.to(device)
        
        outputs = model(img_cuda)
        pred_bbox = outputs['localization'].cpu().squeeze().numpy()
        target_bbox = target_bbox.squeeze().numpy()
        
        # Calculate IoU & Confidence
        iou = calculate_iou(pred_bbox, target_bbox)
        conf = float(torch.softmax(outputs['classification'], dim=1).max().item())

        # Process image for drawing
        display_img = img.squeeze().permute(1,2,0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        display_img = std * display_img + mean
        display_img = np.clip(display_img, 0, 1)
        
        # Convert to PIL Image
        pil_img = Image.fromarray((display_img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)

        # Draw Ground Truth (Green)
        gt_corners = get_corners(target_bbox)
        draw.rectangle(gt_corners, outline="green", width=3)
        draw.text((gt_corners[0], gt_corners[1] - 12), "GT", fill="green")

        # Draw Prediction (Red)
        pred_corners = get_corners(pred_bbox)
        draw.rectangle(pred_corners, outline="red", width=3)
        draw.text((pred_corners[0], pred_corners[1] - 12), "Pred", fill="red")

        # Log to W&B Table - We pass the raw PIL image without W&B's boxes argument
        wb_table.add_data(img_id, wandb.Image(pil_img), round(conf, 4), round(iou, 4))
        count += 1

wandb.log({"Localization_Results": wb_table})
print("Logged 10 images with explicitly colored bounding boxes to W&B.")
wandb.finish()
