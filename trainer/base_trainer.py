import os
import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.torch_utils import get_scheduler
from matplotlib.colors import hsv_to_rgb

class BaseTrainer:

    def __init__(self, cfg, train_loader, valid_loader, logger):
        self.cfg = cfg
        self.cfg_model = cfg.model
        self.cfg_train = cfg.train
        self.cfg_loss = cfg.loss
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.logger = logger

        self.isTrain = self.cfg_train.isTrain
        self.multi_gpu = False
        self.device_ids = self.cfg_train.device_ids
        # self.device = self.device_ids[0]
        self.device = torch.device('cuda:{}'.format(self.device_ids[0])) if self.device_ids else torch.device('cpu')
        # print(self.device)
        self.lr = self.cfg_train.lr

        self.model_names = []
        self.loss_names = []
        self.optimizers = []

        self.save_root = cfg.save_root
        if self.isTrain:
            self.summary_writer = SummaryWriter(str(self.save_root))

        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0

    def _load_networks(self, load_suffix):
        for name in self.model_names:
            load_filename = '{}.pth'.format(name)
            load_path = os.path.join(self.save_root, load_filename)
            state_dict = torch.load(load_path, map_location=str(self.device))
            net = getattr(self, name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

    def _save_networks(self):
        save_suffix = 'epoch_{:<d}'.format(self.i_epoch)
        for name in self.model_names:
            save_filename = '{}.pth'.format(name)
            save_path = os.path.join(self.save_root, save_filename)
            net = getattr(self, name)
            if len(self.device_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda(self.device_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def _print_networks(self, verbose):
        for name in self.model_names:
            net = getattr(self, name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            self.logger.info('[Network {}] Total number of parameters : {:<.3f} M'.format(name, num_params / 1e6))

    def setup(self):
        if self.cfg_train.load_epoch > 0:  # load pretrained model
            load_suffix = 'epoch_{:<d}'.format(self.cfg_train.load_epoch)
            self.logger.info("=> using pre-trained weights {}.".format(load_suffix))
            self._load_networks(load_suffix)

        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, self.cfg_train) for optimizer in self.optimizers]  # schedulers
            self.base_epoch = 0
            if self.cfg_train.continueTrain:
                self.i_epoch = self.cfg_train.load_epoch
            else:
                self.base_epoch = self.cfg_train.load_epoch
            self.update_learning_rate(self.i_epoch if self.i_epoch > 0 else None)

        self._print_networks(self.cfg_train.verbose)  # print network

    def update_learning_rate(self, epoch=None):
        for i, scheduler in enumerate(self.schedulers):
            scheduler.step(epoch)
            c_epoch = scheduler.last_epoch
            c_lr = scheduler.get_lr()

    def mode(self, train=True):
        for name in self.model_names:
            net = getattr(self, name)
            net.train() if train else net.eval()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def resize_flow(self, flow, new_shape):
        flow = flow.unsqueeze(0)
        _, _, h, w = flow.shape
        new_h, new_w = new_shape
        flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                               mode='bilinear', align_corners=True)
        scale_h, scale_w = h / float(new_h), w / float(new_w)
        flow[:, 0] /= scale_w
        flow[:, 1] /= scale_h
        flow = flow.squeeze(0)
        return flow

    def flow_to_image(self, flow, max_flow=256):
        if max_flow is not None:
            max_flow = max(max_flow, 1.)
        else:
            max_flow = np.max(flow)

        n = 8
        u, v = flow[:, :, 0], flow[:, :, 1]
        mag = np.sqrt(np.square(u) + np.square(v))
        angle = np.arctan2(v, u)
        im_h = np.mod(angle / (2 * np.pi) + 1, 1)
        im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
        im_v = np.clip(n - im_s, a_min=0, a_max=1)
        im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
        return (im * 255).astype(np.uint8)

    @abstractmethod
    def _create_loss(self):
        ...

    @abstractmethod
    def _create_optimizer(self):
        ...

    @abstractmethod
    def _create_init_model(self):
        ...

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _test_networks(self):
        ...

    @abstractmethod
    def train(self):
        ...
