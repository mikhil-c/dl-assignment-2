"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, root_dir, split='train'):
        self.img_dir = os.path.join(root_dir, 'images')
        self.anno_dir = os.path.join(root_dir, 'annotations', 'xmls')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        
        self.image_files = []
        for file in os.listdir(self.anno_dir):
            if file.endswith('.xml'):
                base = file[:-4]
                if os.path.exists(os.path.join(self.img_dir, base + '.jpg')):
                    self.image_files.append(base)
                    
        self.image_files = sorted(self.image_files)
        split_idx = int(0.8 * len(self.image_files))
        
        if split == 'train':
            self.image_files = self.image_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
            
        self.classes = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in self.image_files])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            base_name = self.image_files[idx]
            
            img_path = os.path.join(self.img_dir, base_name + '.jpg')
            image = plt.imread(img_path)
            
            if image.ndim == 2:
                image = np.stack((image,)*3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
                
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
                
            class_name = '_'.join(base_name.split('_')[:-1])
            label = self.class_to_idx[class_name]
            
            mask_path = os.path.join(self.mask_dir, base_name + '.png')
            mask = plt.imread(mask_path)
            
            if mask.ndim == 3:
                mask = mask[:, :, 0]
                
            if mask.dtype == np.float32:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
                
            mask = mask - 1
            
            xml_path = os.path.join(self.anno_dir, base_name + '.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bbox_elem = root.find('object').find('bndbox')
            
            xmin = float(bbox_elem.find('xmin').text)
            ymin = float(bbox_elem.find('ymin').text)
            xmax = float(bbox_elem.find('xmax').text)
            ymax = float(bbox_elem.find('ymax').text)
            
            transformed = self.transform(
                image=image, 
                bboxes=[[xmin, ymin, xmax, ymax]], 
                class_labels=[label], 
                mask=mask
            )
            
            image_tensor = transformed['image']
            mask_tensor = torch.tensor(transformed['mask'], dtype=torch.long)
            
            aug_bbox = transformed['bboxes'][0]
            x_center = (aug_bbox[0] + aug_bbox[2]) / 2.0
            y_center = (aug_bbox[1] + aug_bbox[3]) / 2.0
            width = aug_bbox[2] - aug_bbox[0]
            height = aug_bbox[3] - aug_bbox[1]
            
            bbox_tensor = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return image_tensor, label_tensor, bbox_tensor, mask_tensor
            
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
