import os
import json
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.ops.boxes import box_convert
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

MAX_IMG_SIZE = 224


def parse_images_and_bboxes(image_dir, mask2face=False):
    """
    Parse a directory with images.
    :param image_dir: Path to directory with images.
    :param mask2face: if True move the bbox to cover the upper face
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    for filename in example_filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")

        img = Image.open(os.path.join(image_dir, filename)).convert('RGB')
        bbox = json.loads(bbox)
        x, y, w, h = bbox
        if not (x >= 0 and y >= 0 and h > 0 and w > 0):
            print(f'{filename} has faulty bbox and will be ignored')
            continue
        if mask2face:
            bbox = mask2face_bbox(bbox)
        bbox = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)
        bbox = box_convert(bbox, in_fmt='xywh', out_fmt='xyxy')
        img = T.ToTensor()(img)
        proper_mask = torch.tensor([1 if proper_mask.lower() == "true" else 2])
        data.append((filename, image_id, img, bbox, proper_mask))
    return data


def mask2face_bbox(bbox, include_mask=False):
    x, y, w, h = bbox
    new_h = h//2
    y = max(0, y - new_h)
    if include_mask:
        new_h += h
    return [x, y, w, new_h]


def calc_iou(bbox_a, bbox_b, bbox_format='xyxy'):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    if bbox_format=='xyxy':
        bbox_a = box_convert(bbox_a, in_fmt='xyxy', out_fmt='xywh')
        bbox_b = box_convert(bbox_b, in_fmt='xyxy', out_fmt='xywh')
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection  # Union = Total Area - Intersection
    return intersection / union


def calc_iou_batch(bboxes_a, bboxes_b, bbox_format='xyxy'):
    iou_list = []
    for bbox_a, bbox_b in zip(bboxes_a, bboxes_b):
        if bbox_a is None or bbox_b is None:
            iou_list.append(0)
            continue
        iou = calc_iou(bbox_a, bbox_b, bbox_format)
        iou_list.append(iou)
    return np.mean(iou_list)


def plot_image_and_bbox(im, bbox_true, bbox_prediction):
    fig, ax = plt.subplots()
    ax.imshow(im)
    for bbox, label, color in zip([bbox_true, bbox_prediction], ['label', 'prediction'], ['g', 'b']):
        bbox_xywh = box_convert(bbox, in_fmt='xyxy', out_fmt='xywh')
        x1, y1, w1, h1 = bbox_xywh[0]
        rect = patches.Rectangle((x1, y1), w1, h1,
                                linewidth=2, edgecolor=color, facecolor='none', label=label)
        ax.add_patch(rect)
    ax.axis('off')
    fig.legend()
    plt.show()


def random_bbox_from_image(img):
    """
    Randomly picks a bounding box given an image.
    :param bbox: Iterable with numbers.
    :param image: PIL image
    :return: Random bounding box, relative to the input image.
    """
    H, W = img.height, img.width
    y = np.random.randint(2, H-1)
    x = np.random.randint(2, W-1)
    h = np.random.randint(1, H-y-1)
    w = np.random.randint(1, W-x-1)
    return [x, y, w, h]


