import torch
import wandb
import numpy as np
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from torchvision import transforms
from models.multitask import MultiTaskPerceptionModel

# 1. Initialize W&B
wandb.init(project="da6401-assignment-2", name="task-2.7")

# 2. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskPerceptionModel().to(device)
model.eval()

# 3. Setup Class Names (Standard Oxford-IIIT Pet Classes)
class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'giant_schnauzer', 'golden_retriever', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

# 4. Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. URLs
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Rottweiler_standing_facing_left.jpg/1200px-Rottweiler_standing_facing_left.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Retriever_in_water.jpg/1200px-Retriever_in_water.jpg"
]

headers = {'User-Agent': 'Mozilla/5.0'}

def get_corners(box):
    return [box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2]

# 6. Pipeline Execution
columns = ["Image_Source", "Pipeline_Output", "Predicted_Breed", "Confidence"]
wb_table = wandb.Table(columns=columns)

for i, url in enumerate(urls):
    try:
        response = requests.get(url, headers=headers, timeout=15)
        raw_img = Image.open(BytesIO(response.content)).convert("RGB")
        input_tensor = preprocess(raw_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # 1. Localization
        pred_bbox = outputs['localization'].cpu().squeeze().numpy()
        pred_corners = get_corners(pred_bbox)
        
        # 2. Classification
        probs = torch.softmax(outputs['classification'], dim=1)
        conf, class_idx = torch.max(probs, dim=1)
        breed = class_names[class_idx.item()]
        
        # 3. Segmentation
        seg_mask = torch.argmax(outputs['segmentation'], dim=1).cpu().squeeze().numpy()
        
        # Visualization
        disp_img = raw_img.resize((224, 224))
        draw = ImageDraw.Draw(disp_img)
        draw.rectangle(pred_corners, outline="red", width=3)
        draw.text((pred_corners[0], pred_corners[1]-12), f"{breed}", fill="red")
        
        wb_table.add_data(
            f"Novel_{i+1}",
            wandb.Image(disp_img, masks={"prediction": {"mask_data": seg_mask}}),
            breed,
            round(conf.item(), 4)
        )
    except Exception as e:
        print(f"Error on {i}: {e}")

wandb.log({"Final_Pipeline_Showcase": wb_table})
wandb.finish()
