import torch
from torch.utils.data import DataLoader
from dataset import FaceMaskDataset, collate_fn
from custom_rcnn import custom_fasterrcnn_resnet50_fpn
from trainer import Trainer

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Setting up GPU device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training on {device}')

    # Setting up the model
    model = custom_fasterrcnn_resnet50_fpn()
    model = model.to(device)

    dataset_train = FaceMaskDataset('../train')
    dataset_test = FaceMaskDataset('../test')

    train_data_loader = DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_data_loader = DataLoader(
        dataset_test, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Setting the optimizer, epochs
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.001,
    #                             momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 20

    checkpoint = None
    trainer = Trainer(model, optimizer, lr_scheduler, device)
    trainer.fit(train_data_loader, test_data_loader, num_epochs, save_checkpoint=True, checkpoint_file=checkpoint)
