import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.listloss = nn.MSELoss()
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if isinstance(prediction, list):
            loss = torch.FloatTensor([0]).cuda()

            if self.gan_mode in ['lsgan', 'vanilla']:
                for prediction_i in prediction:
                    target_tensor = self.get_target_tensor(prediction_i, target_is_real)
                    loss += self.loss(prediction_i, target_tensor.cuda())
                return loss / 32

            elif self.gan_mode == 'wgangp':
                for prediction_i in prediction:
                    loss += -prediction_i.mean()
                return loss / 32
        else:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor.cuda())
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            return loss

