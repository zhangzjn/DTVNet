import copy
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from .dtv_datasets import SkyTimelapse

def get_dataset(cfg):
    cfg_data, cfg_data_aug, cfg_train = cfg.data, cfg.data_aug, cfg.train
    train_loader = list()
    valid_loader = list()
    if cfg_data.type == 'sky_timelapse':
        # train set
        transform = transforms.Compose([transforms.Resize((cfg_data_aug.imageSize, cfg_data_aug.imageSize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        train_set = SkyTimelapse(cfg_data.root, cfg_data.split_train, nframes=cfg_data.nframes, transform=transform)
        train_loader = DataLoader(train_set,
                                  batch_size=cfg_train.batch_size,
                                  num_workers=cfg_train.num_workers,
                                  shuffle=cfg_train.shuffle,
                                  drop_last=cfg_train.drop_last,
                                  pin_memory=cfg_train.pin_memory)

        # validation set
        transform = transforms.Compose([transforms.Resize((cfg_data_aug.imageSize, cfg_data_aug.imageSize)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        valid_set = SkyTimelapse(cfg_data.root, cfg_data.split_valid, nframes=cfg_data.nframes, transform=transform)
        valid_loader = DataLoader(valid_set,
                                  batch_size=cfg_train.batch_size,
                                  num_workers=cfg_train.num_workers,
                                  shuffle=False,
                                  drop_last=False,
                                  pin_memory=cfg_train.pin_memory)

    else:
        raise NotImplementedError(cfg_data.type)
    return train_loader, valid_loader
