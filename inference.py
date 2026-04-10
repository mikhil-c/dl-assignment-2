"""Inference and evaluation
"""

import torch
import numpy as np
from sklearn.metrics import f1_score
from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from torch.utils.data import DataLoader

def calculate_iou(pred_box: torch.Tensor, target_box: torch.Tensor) -> float:
    pred_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
    pred_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
    pred_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
    pred_y2 = pred_box[:, 1] + pred_box[:, 3] / 2

    tgt_x1 = target_box[:, 0] - target_box[:, 2] / 2
    tgt_y1 = target_box[:, 1] - target_box[:, 3] / 2
    tgt_x2 = target_box[:, 0] + target_box[:, 2] / 2
    tgt_y2 = target_box[:, 1] + target_box[:, 3] / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    pred_area = pred_box[:, 2] * pred_box[:, 3]
    tgt_area = target_box[:, 2] * target_box[:, 3]
    union_area = pred_area + tgt_area - inter_area + 1e-6

    iou = inter_area / union_area
    return iou.mean().item()

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

def evaluate(device: torch.device, test_loader: DataLoader):
    model = MultiTaskPerceptionModel().to(device)
    model.eval()

    all_preds = []
    all_targets = []
    total_iou = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, labels, bboxes, masks in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            masks = masks.to(device)

            outputs = model(images)
            
            class_logits = outputs['classification']
            pred_bboxes = outputs['localization']
            pred_masks = outputs['segmentation']

            preds = torch.argmax(class_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            total_iou += calculate_iou(pred_bboxes, bboxes)
            total_dice += calculate_dice(pred_masks, masks)

    f1 = f1_score(all_targets, all_preds, average='macro')
    mean_iou = total_iou / len(test_loader)
    mean_dice = total_dice / len(test_loader)

    print(f"Classification Macro F1: {f1:.4f}")
    print(f"Localization Mean Box IoU: {mean_iou:.4f}")
    print(f"Segmentation Dice Score: {mean_dice:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = OxfordIIITPetDataset(root_dir="data", split='test')
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    evaluate(device, test_loader)
