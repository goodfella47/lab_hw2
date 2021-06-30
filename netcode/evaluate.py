import os
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.ops.boxes import box_convert
from custom_rcnn import custom_fasterrcnn_resnet50_fpn
from utils import random_bbox_from_image


def get_predictions(image_dir, data, saved_state):
    bbox_pred, proper_mask_pred = [], []

    torch.multiprocessing.set_sharing_strategy('file_system')

    # Setting up GPU device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Evaluate on {device}')
    # Setting up the model

    model = custom_fasterrcnn_resnet50_fpn()
    model.load_state_dict(torch.load(saved_state, map_location=device)["model_state"])
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for d in tqdm(data):
            img = Image.open(os.path.join(image_dir, d[0])).convert('RGB')
            img_tensor = [T.ToTensor()(img).to(device)]
            _, prediction = model(img_tensor)

            try:
                bbox = prediction[0]['boxes'][0].unsqueeze(0).detach().cpu().int().tolist()
                proper_mask = prediction[0]['labels'][0].item()
                proper_mask = True if proper_mask == 1 else False
            except:
                bbox = random_bbox_from_image(img)
                proper_mask = False

            bbox_pred.append(bbox)
            proper_mask_pred.append(proper_mask)

    return bbox_pred, proper_mask_pred

# if __name__ == "__main__":
#     torch.multiprocessing.set_sharing_strategy('file_system')
#
#     # Setting up GPU device
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(f'Evaluate on {device}')
#     # Setting up the model
#
#     model = custom_fasterrcnn_resnet50_fpn()
#     model.load_state_dict(torch.load('model_epoch_2_loss_0.134.pt', map_location=device)["model_state"])
#     model = model.to(device)
#
#     dataset_test = FaceMaskDataset('../test')
#
#     test_data_loader = DataLoader(
#         dataset_test, batch_size=5, shuffle=True, num_workers=0, collate_fn=collate_fn)
#
#     # Main training function
#     model.eval()
#     loss_list = []
#
#     for images, targets in tqdm(test_data_loader):
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) if k == 'labels' else v.float().to(device) for k, v in t.items()} for t in
#                    targets]
#
#         _, predictions = model(images)
#         if predictions[0]['boxes'].numel() > 0:
#             predicted_box = predictions[0]['boxes'][0].unsqueeze(0).detach().cpu()
#             print('score: ', predictions[0]['scores'][0])
#             print('prediction bbox: ', predicted_box)
#             print(targets[0]['boxes'].detach().cpu())
#             plot_image_and_bbox(images[0].permute(1, 2, 0).detach().cpu(), targets[0]['boxes'].detach().cpu(),
#                                 predicted_box)

