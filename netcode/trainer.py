import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from utils import calc_iou_batch


class Trainer:
    def __init__(self, model, optimizer, lr_scheduler=None, device="cpu", resize_bbox=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.resize_bbox = resize_bbox
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader, num_epochs,
            plot_image_after_epoch=False, save_checkpoint=True, checkpoint_file=None):
        train_loss, test_loss, train_acc, test_acc, train_iou, test_iou = [], [], [], [], [], []
        if checkpoint_file is not None:
            if os.path.isfile(checkpoint_file):
                print(f"*** Loading checkpoint file {checkpoint_file}")
                saved_state = torch.load(checkpoint_file, map_location=self.device)
                train_loss = saved_state['train_loss']
                test_loss = saved_state['test_loss']
                train_acc = saved_state['train_acc']
                test_acc = saved_state['test_acc']
                train_iou = saved_state['train_iou']
                test_iou = saved_state['test_iou']
                self.model.load_state_dict(saved_state["model_state"])
        for epoch in range(num_epochs):
            predicted_boxes = None
            self.model.train()
            epoch_loss_list, epoch_iou_list, epoch_acc_list = [], [], []
            start = time.time()

            # train batch
            for images, targets in tqdm(dl_train):
                loss_batch, batch_iou, batch_acc = self.train_batch(images, targets)
                epoch_loss_list.append(loss_batch)
                epoch_iou_list.append(batch_iou)
                epoch_acc_list.append(batch_acc)

            end = time.time()
            epoch_loss = np.mean(epoch_loss_list)
            epoch_iou = np.mean(epoch_iou_list)
            epoch_acc = np.mean(epoch_acc_list)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(
                f'Epoch {epoch+1} loss: {epoch_loss:.3f}, iou: {epoch_iou:.3f}, accuracy: {epoch_acc:.3f}, time taken: ({end - start:.1f}s)')

            epoch_loss_test, epoch_iou_test, epoch_acc_test = self.evaluate(dl_test)
            print(
                f'Test {epoch+1} loss: {epoch_loss_test:.3f}, iou: {epoch_iou_test:.3f}, accuracy: {epoch_acc_test:.3f}')

            train_loss.append(epoch_loss)
            train_iou.append(epoch_iou)
            train_acc.append(epoch_acc)
            test_loss.append(epoch_loss_test)
            test_iou.append(epoch_iou_test)
            test_acc.append(epoch_acc_test)

            if save_checkpoint:
                checkpoint_filename = f'model_epoch_{epoch + 1}_loss_{epoch_loss:.3}.pt'
                saved_state = dict(
                    train_loss=train_loss,
                    train_iou=train_iou,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_iou=test_iou,
                    test_acc=test_acc,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch + 1}"
                )
        return train_loss, train_iou, train_acc, test_loss, test_iou, test_acc

    def train_batch(self, images, targets):
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) if k == 'labels' else v.float().to(self.device) for k, v in t.items()}
                   for t in targets]
        losses_dict, detections = self.model(images, targets)
        losses = sum(loss for loss in losses_dict.values())
        loss_value = losses.item()

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        # calculate iou
        predicted_boxes = [detected['boxes'][0].detach().cpu() if detected['boxes'].nelement() > 0 else None for
                           detected in detections]
        target_boxes = [t['boxes'][0].cpu() for t in targets]
        batch_iou = calc_iou_batch(predicted_boxes, target_boxes)

        # calculate accuracy
        predicted_labels = torch.tensor([detected['labels'][0].item() if detected['boxes'].nelement() > 0 else 1 for
                                         detected in detections])
        target_labels = torch.tensor([t['labels'][0].item() for t in targets])
        batch_acc = torch.sum(predicted_labels == target_labels) / len(target_labels)

        return loss_value, batch_iou, batch_acc

    def test_batch(self, images, targets):
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) if k == 'labels' else v.float().to(self.device) for k, v in t.items()}
                   for t in targets]

        with torch.no_grad():
            losses_dict, detections = self.model(images, targets)
            losses = sum(loss for loss in losses_dict.values())
            loss_value = losses.item()

            # calculate iou
            predicted_boxes = [detected['boxes'][0].cpu() if detected['boxes'].nelement() > 0 else None for
                               detected in detections]
            target_boxes = [t['boxes'][0].cpu() for t in targets]
            batch_iou = calc_iou_batch(predicted_boxes, target_boxes)

            # calculate accuracy
            predicted_labels = torch.tensor([detected['labels'][0].item() if detected['boxes'].nelement() > 0 else 2 for
                                             detected in detections])
            target_labels = torch.tensor([t['labels'][0].item() for t in targets])
            batch_acc = torch.sum(predicted_labels == target_labels) / len(target_labels)

        return loss_value, batch_iou, batch_acc

    def evaluate(self, dl_test):
        self.model.eval()
        loss_list, iou_list, acc_list = [], [], []
        for images, targets in tqdm(dl_test):
            loss_batch, batch_iou, batch_acc = self.test_batch(images, targets)
            loss_list.append(loss_batch)
            iou_list.append(batch_iou)
            acc_list.append(batch_acc)
        return np.mean(loss_list), np.mean(iou_list), np.mean(acc_list)
