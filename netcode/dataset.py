from torch.utils.data import Dataset
from utils import parse_images_and_bboxes
import torchvision.transforms as T

class FaceMaskDataset(Dataset):
    def __init__(self, image_dir, mask2face=False):
        self.image_dir = image_dir
        self.img_data = parse_images_and_bboxes(self.image_dir, mask2face=mask2face)

    def __getitem__(self, idx):
        filename, _, img, bbox, proper_mask = self.img_data[idx]
        target = {
            "boxes": bbox,
            "labels": proper_mask
        }
        return img, target

    def __len__(self):
        return len(self.img_data)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))
