import os
import time
import torch
import numpy as np
from .base_trainer import BaseTrainer
from lossoptim.loss import GANLoss
from lossoptim.wgangp import cal_gradient_penalty
from lossoptim.optim import AdamW
from utils.misc_utils import AverageMeter, mkdirs
from utils.torch_utils import init_weights
from models.dtv import DTVG, DTVD
import torchvision.utils as vutils
from models.flowencoder import FlowEncoder
from collections import OrderedDict
import imageio


class TrainFramework(BaseTrainer):
    def __init__(self, cfg, train_loader, valid_loader, logger):
        super(TrainFramework, self).__init__(cfg, train_loader, valid_loader, logger)

        self.model_names = ['DTV_G']
        if self.isTrain:
            self.model_names.append('DTV_D')

        self._create_init_model()  # model
        if self.isTrain:
            self._create_loss()  # loss
            self._create_optimizer()


    def _create_init_model(self):
        self.logger.info("=> create and init models [{}] with {}".format(
            ', '.join(self.model_names), self.cfg_model.init_type))
        self.DTV_G = DTVG(ngf=self.cfg_model.ngf, dlatent_size=self.cfg_model.dlatent_size, n_blocks=self.cfg_model.n_blocks, 
                          use_2d=self.cfg_model.use_2d, use_flow=self.cfg_model.use_flow).to(self.device)
        self.DTV_G = torch.nn.DataParallel(self.DTV_G, device_ids=self.device_ids)
        init_weights(self.DTV_G, init_type=self.cfg_model.init_type, init_gain=self.cfg_model.init_gain)
        if self.isTrain:
            self.DTV_D = DTVD(self.cfg_model.ndf).to(self.device)
            self.DTV_D = torch.nn.DataParallel(self.DTV_D, device_ids=self.device_ids)
            init_weights(self.DTV_D, init_type=self.cfg_model.init_type, init_gain=self.cfg_model.init_gain)
        
        self.Flow_G = FlowEncoder('Flow', self.cfg_model.ndf, self.cfg_model.dlatent_size).to(self.device)
        self.Flow_G = torch.nn.DataParallel(self.Flow_G, device_ids=self.device_ids)
        

    def _create_loss(self):
        if self.cfg_loss.RGB_L1:
            self.criterionRGBL1 = torch.nn.L1Loss()
            self.loss_names.append('RGB_L1')
        if self.cfg_loss.Flow_L1:
            self.criterionFlowL1 = torch.nn.L1Loss()
            self.loss_names.append('Flow_L1')
        if self.cfg_loss.gan:
            self.criterionGAN = GANLoss(self.cfg_loss.gan_mode)
            self.loss_names.append('gan')
        self.logger.info("=> create losses [{}]".format(', '.join(self.loss_names)))


    def _create_optimizer(self):
        self.logger.info('=> setting {} optimizer'.format(self.cfg_train.optim))
        if self.cfg_train.optim == 'adam':
            self.optimizer_DTV_G = torch.optim.Adam(self.DTV_G.parameters(), lr=self.cfg_train.lr_DTV_G,
                                                   betas=(self.cfg_train.momentum, self.cfg_train.beta),
                                                   eps=self.cfg_train.eps)
            self.optimizer_DTV_D = torch.optim.Adam(self.DTV_D.parameters(), lr=self.cfg_train.lr_DTV_D,
                                                   betas=(self.cfg_train.momentum, self.cfg_train.beta),
                                                   eps=self.cfg_train.eps)
            self.optimizer_FE = torch.optim.Adam(self.Flow_G.parameters(), lr=self.cfg_train.lr,
                                                 betas=(self.cfg_train.momentum, self.cfg_train.beta),
                                                 eps=self.cfg_train.eps)
            self.optimizers.append(self.optimizer_DTV_G)
            self.optimizers.append(self.optimizer_DTV_D)
            self.optimizers.append(self.optimizer_FE)

        elif self.cfg_train.optim == 'adamw':
            self.optimizer_DTV_G = AdamW(self.DTV_G.parameters(), lr=self.cfg_train.lr,
                                        betas=(self.cfg_train.momentum, self.cfg_train.beta),
                                        eps=self.cfg_train.eps)
            self.optimizer_DTV_D = AdamW(self.DTV_D.parameters(), lr=self.cfg_train.lr,
                                        betas=(self.cfg_train.momentum, self.cfg_train.beta),
                                        eps=self.cfg_train.eps)
            self.optimizers.append(self.optimizer_DTV_G)
            self.optimizers.append(self.optimizer_DTV_D)

    def set_input(self, input):
        self.frames_img_gt, self.frames_flow_in, self.frames_flow_gt, self.frames_A, self.cls = input
        # self.frames_A = self.frames_A.unsqueeze(2).repeat(1, 1, 32, 1, 1)
        self.frames_A = self.frames_A.to(self.device)
        self.frames_flow_gt = self.frames_flow_gt.to(self.device)
        self.frames_flow_in = self.frames_flow_in.to(self.device)
        self.frames_img_gt = self.frames_img_gt.to(self.device)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()
        key_meter_names = ['G_RGB_L1', 'G_Flow_L1', 'GP', 'G_GAN', 'D_real', 'D_fake']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4, names=key_meter_names)
        self.mode(train=True)
        
        t1 = time.time()  # time -> base
        for i_step, data in enumerate(self.train_loader):
            self.set_input(data)
            t2 = time.time()  # time -> load data
            self.optimize_parameters()
            t3 = time.time()  # time -> forward and backward

            am_data_time.update(t2 - t1)  # update meters
            am_batch_time.update(t3 - t2)
            key_meters.update([self.loss_G_RGB_L1.item(), self.loss_G_Flow_L1.item(), self.loss_gp.item(),
                               self.loss_G_GAN.item(), self.loss_D_real.item(), self.loss_D_fake.item()
                               ], n=1)
            # key_meters.update([self.loss_G_L1.item(), self.loss_G_L1.item(), self.loss_G_L1.item(), self.loss_G_L1.item()], n=1)

            if self.i_iter % self.cfg_train.print_frep == 0:  # update meters
                for name, v in zip(key_meter_names, key_meters.val):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)
            t4 = time.time()
            if self.i_iter % self.cfg_train.record_freq == 0:
                str = '{:>3d}({}):{:>3d}/{:<3d}\t'.format(self.i_epoch + 1, self.base_epoch, i_step, len(self.train_loader)) + \
                       'Time {} Data {}\t'.format(am_batch_time, am_data_time) + \
                       'Losses {}'.format(key_meters)
                self.logger.info(str)
            t1 = time.time()
            self.i_iter += 1
        self.i_epoch += 1

    def train(self):
        self.setup()
        for epoch in range(self.cfg_train.epoch_count, self.cfg_train.niter + self.cfg_train.niter_decay):
            self._run_one_epoch()

            if self.i_epoch % self.cfg_train.save_epoch == 0:
                self._save_networks()
                self.logger.info(' * save model (epoch {}) '.format(self.i_epoch))

            if self.i_epoch % self.cfg_train.test_epoch == 0:
                self._test_networks(self.i_epoch)
                self.logger.info(' * test model (epoch {}) '.format(self.i_epoch))

            self.update_learning_rate()

    def forward(self):
        self.latent = self.Flow_G(self.frames_flow_in)
        self.frames_img_f, self.frames_flow_f = self.DTV_G(self.frames_A, self.latent)

    def backward_DTV_D(self):

        pred_fake = self.DTV_D(self.frames_img_f.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)  # Fake

        pred_real = self.DTV_D(self.frames_img_gt)
        self.loss_D_real = self.criterionGAN(pred_real, True)  # Real

        # wgan-gp
        self.loss_gp, gradients = cal_gradient_penalty(self.DTV_D, self.frames_img_gt, self.frames_img_f, self.device, lambda_gp=10.0)
        self.loss_gp.backward(retain_graph=True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.cfg_loss.lambda_D  # combine loss and calculate gradients
        self.loss_D.backward()

    def backward_DTV_G(self):
        pred_fake = self.DTV_D(self.frames_img_f)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_RGB_L1 = self.criterionRGBL1(self.frames_img_f, self.frames_img_gt) * self.cfg_loss.lambda_L1_RGB  # loss -> L1
        self.loss_G_Flow_L1 = self.criterionFlowL1(self.frames_flow_f, self.frames_flow_gt) * self.cfg_loss.lambda_L1_Flow  # loss -> L1

        self.loss_G = self.loss_G_GAN + self.loss_G_RGB_L1 + self.loss_G_Flow_L1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.DTV_D, True)  # enable backprop for D
        self.optimizer_DTV_D.zero_grad()     # set D's gradients to zero
        self.backward_DTV_D()                # calculate gradients for D
        self.optimizer_DTV_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.DTV_D, False)  # D requires no gradients when optimizing G
        self.optimizer_DTV_G.zero_grad()        # set G's gradients to zero
        self.optimizer_FE.zero_grad()
        self.backward_DTV_G()                   # calculate graidents for G
        self.optimizer_DTV_G.step()             # udpate G's weights
        self.optimizer_FE.step()


    @torch.no_grad()
    def _test_networks(self, epoch):
        def generate_video(path_in, path_out, name):
            img_path = path_in + '/%04d.png'
            mp4_path = '{}/{}.mp4'.format(path_out, path_in.split('/')[-1])
            cmd = (
                        'ffmpeg -loglevel warning -framerate 25 -pix_fmt yuv420p -i ' + img_path + ' -qscale:v 2 -y ' + mp4_path)
            os.system(cmd)
            
        self.mode(train=False)
        save_result_dir = '{}/results'.format(self.save_root)
        save_image_dir_f = '{}/{}_f'.format(save_result_dir, epoch)
        save_image_dir_r = '{}/{}_r'.format(save_result_dir, epoch)
        save_video_dir_f = '{}/{}_f_v'.format(save_result_dir, epoch)
        save_video_dir_r = '{}/{}_r_v'.format(save_result_dir, epoch)
        
        mkdirs([save_result_dir, save_image_dir_f, save_image_dir_r, save_video_dir_f, save_video_dir_r])

        for i_step, data in enumerate(self.valid_loader):

            val_img_gt, val_flow_in, val_flow_gt, val_in, val_cls = data

            val_img_gt = val_img_gt.to(self.device)
            val_flow_in = val_flow_in.to(self.device)
            val_flow_gt = val_flow_gt.to(self.device)
            val_in = val_in.to(self.device)
            
            t1 = time.time()
            latent = self.Flow_G(val_flow_in)
            #latent = torch.randn(1,512)
            torch.cuda.synchronize()
            t2 = time.time()
            val_fake_img, val_fake_flow = self.DTV_G(val_in, latent)
            t3 = time.time()
            torch.cuda.synchronize()
            val_fake_img = val_fake_img.data.permute(2, 0, 1, 3, 4)

            for b in range(val_fake_img.size(1)):
                save_image_dir_f_s = '{}/{}_f/{}'.format(save_result_dir, epoch, val_cls[b])
                mkdirs(save_image_dir_f_s)
                for t in range(val_fake_img.size(0)):
                    vutils.save_image(val_fake_img[t][b], '{}/{:>04d}.png'.format(save_image_dir_f_s, t), normalize=True, nrow=1)
                generate_video(save_image_dir_f_s, save_video_dir_f, val_cls[b])

            val_img_gt = val_img_gt.permute(2, 0, 1, 3, 4)

            for b in range(val_img_gt.size(1)):
                save_image_dir_r_s = '{}/{}_r/{}'.format(save_result_dir, epoch, val_cls[b])
                mkdirs(save_image_dir_r_s)
                for t in range(val_img_gt.size(0)):
                    vutils.save_image(val_img_gt[t][b], '{}/{:>04d}.png'.format(save_image_dir_r_s, t), normalize=True, nrow=1)
                generate_video(save_image_dir_r_s, save_video_dir_r, val_cls[b])

    def test(self):
        self.setup()
        self._test_networks('test_{}'.format(self.cfg_train.load_epoch))

