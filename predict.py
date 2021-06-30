import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.ops.boxes import box_convert
from tqdm import tqdm
from netcode.custom_rcnn import custom_fasterrcnn_resnet50_fpn
from netcode.utils import random_bbox_from_image

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
model_path = 'model/model_final.pt'
bbox_pred = []
proper_mask_pred = []

# Setting up device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Predicting on {device}')

# Setting up the model
model = custom_fasterrcnn_resnet50_fpn()
model.load_state_dict(torch.load(model_path, map_location=device)["model_state"])
model = model.to(device)

model.eval()
with torch.no_grad():
    for file_name in tqdm(files):
        img = Image.open(os.path.join(args.input_folder, file_name)).convert('RGB')
        img_tensor = [T.ToTensor()(img).to(device)]
        _, prediction = model(img_tensor)

        try:
            bbox_xyxy = prediction[0]['boxes'][0].unsqueeze(0).detach().cpu()
            bbox = box_convert(bbox_xyxy, in_fmt='xyxy', out_fmt='xywh').int().tolist()[0]
            proper_mask = prediction[0]['labels'][0].item()
            proper_mask = True if proper_mask == 1 else False
        except:
            bbox = random_bbox_from_image(img)
            proper_mask = False

        bbox_pred.append(bbox)
        proper_mask_pred.append(proper_mask)

prediction_df = pd.DataFrame(zip(files, *np.array(bbox_pred).T, proper_mask_pred),
                             columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
prediction_df.to_csv("prediction.csv", index=False, header=True)
