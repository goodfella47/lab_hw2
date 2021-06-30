import os
import numpy as np
import torch
from torchvision.ops.boxes import box_convert
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
import random

np.random.seed(42)
image_dir = "example_images"


def parse_images_and_bboxes(image_dir, image_num=None, shuffle=False):
    """
    Parse a directory with images.
    :param image_dir: Path to directory with images.
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    for filename in example_filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)
        bbox = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)
        bbox = box_convert(bbox, in_fmt='xywh', out_fmt='xyxy').tolist()
        proper_mask = True if proper_mask.lower() == "true" else False
        data.append((filename, image_id, bbox, proper_mask))
    if shuffle is not None:
        random.shuffle(data)
    if image_num:
        data = data[:image_num]
    return data


def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union


def show_images_and_bboxes(data, image_dir):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    for filename, image_id, bbox, proper_mask in data:
        im = cv2.imread(os.path.join(image_dir, filename))
        im = im[:, :, ::-1]
        x1, y1, w1, h1 = bbox
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        fig.suptitle(f"proper_mask={proper_mask}")
        ax.axis('off')
        plt.show()


def get_images_and_labels(data, image_dir):
    images = []
    labels = []
    for filename, image_id, bbox, proper_mask in data:
        im = cv2.imread(os.path.join(image_dir, filename))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = [int(p) for p in bbox[0]]
        start_point = (x1, y1)
        end_point = (x2, y2)
        im = cv2.rectangle(im, start_point, end_point, (0, 255, 0), 2)
        images.append(im)
        labels.append(proper_mask)
    return images, labels


def get_images_and_labels_on_eval(data, bbox_pred_list, proper_mask_pred_list, image_dir):
    images = []
    labels = []
    for (filename, image_id, bbox, proper_mask), bbox_pred, proper_mask_pred in zip(data, bbox_pred_list, proper_mask_pred_list):
        im = cv2.imread(os.path.join(image_dir, filename))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # draw true bbox
        x1, y1, x2, y2 = [int(p) for p in bbox[0]]
        start_point = (x1, y1)
        end_point = (x2, y2)
        im = cv2.rectangle(im, start_point, end_point, (0, 255, 0), 2)

        # draw predicted bbox
        x1, y1, x2, y2 = [int(p) for p in bbox_pred[0]]
        start_point = (x1, y1)
        end_point = (x2, y2)
        im = cv2.rectangle(im, start_point, end_point, (255, 0, 0), 2)

        labal = f'Label: {proper_mask}, Predicted: {proper_mask_pred}'

        images.append(im)
        labels.append(labal)

    return images, labels


def random_bbox_predict(bbox):
    """
    Randomly predicts a bounding box given a ground truth bounding box.
    For example purposes only.
    :param bbox: Iterable with numbers.
    :return: Random bounding box, relative to the input bbox.
    """
    return [x + np.random.randint(-15, 15) for x in bbox]


if __name__ == "__main__":
    data = parse_images_and_bboxes(image_dir)
    show_images_and_bboxes(data, image_dir)
