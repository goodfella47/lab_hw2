import os
import argparse
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.ops.boxes import box_convert
from netcode.net import custom_fasterrcnn_resnet50_fpn

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
bbox_pred = []
proper_mask_pred = []

# Setting up device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Predicting on {device}')

# Setting up the model
model = custom_fasterrcnn_resnet50_fpn()
model.load_state_dict(torch.load('model_epoch_2_loss_0.134.pt', map_location=device)["model_state"])
model = model.to(device)

model.eval()
for file_name in files:
    img = Image.open(os.path.join(args.input_folder, file_name)).convert('RGB')
    img = [T.ToTensor()(img)]
    _, prediction = model(img)

    bbox_xyxy = prediction[0]['boxes'][0].unsqueeze(0)
    bbox = box_convert(bbox_xyxy, in_fmt='xyxy', out_fmt='xywh')
    proper_mask = prediction[0]['labels'][0].item()

    bbox_pred.append(bbox)
    proper_mask_pred.append(proper_mask)

prediction_df = pd.DataFrame(zip(files, *bbox_pred, proper_mask_pred),
                             columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
prediction_df.to_csv("prediction.csv", index=False, header=True)
