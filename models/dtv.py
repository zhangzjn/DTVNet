import torch
import torch.nn  as nn
from torch.nn import init
from math import sqrt
import functools
from lossoptim.loss import GANLoss

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class G_synthesis(nn.Module):
    def __init__(self, ngf, dlatent_size, n_blocks=6):
        super(G_synthesis, self).__init__()
        self.ngf = ngf
        self.dlatent_size = dlatent_size
        self.n_blocks = n_blocks
        # solve input image
        conv = [nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)]
        self.conv = nn.Sequential(*conv)
        # solve input vector
        n_mlp = 3
        opt = []
        for i in range(n_mlp):
            opt.append(EqualLinear(dlatent_size, dlatent_size))
            opt.append(nn.LeakyReLU(0.2))

        model_img = []
        model_A = []
        model_AdaIN = []
        for i in range(n_blocks):
            model_img += [ResnetBlock(ngf, norm_layer=nn.BatchNorm2d, use_bias=False)]
            model_A += [nn.Sequential(EqualLinear(dlatent_size, dlatent_size), nn.LeakyReLU(0.2),
                                      EqualLinear(dlatent_size, dlatent_size), nn.LeakyReLU(0.2))]
            model_AdaIN += [AdaptiveInstanceNorm(ngf, dlatent_size)]

        model_opt = [nn.Conv2d(ngf, 2 * 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
                     nn.LeakyReLU(0.1, inplace=True)]

        self.opt = nn.Sequential(*opt)
        self.model_img = nn.Sequential(*model_img)
        self.model_A = nn.Sequential(*model_A)
        self.model_AdaIN = nn.Sequential(*model_AdaIN)
        self.model_opt = nn.Sequential(*model_opt)

    def forward(self, x, dlatent):
        x = self.conv(x)
        opt = self.opt(dlatent)
        for i in range(self.n_blocks):
            x = self.model_img[i](x)
            opt_sep = self.model_A[i](opt)
            x = self.model_AdaIN[i](x, opt_sep)
        out = self.model_opt(x)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

    def build_conv_block(self, dim, norm_layer, use_bias):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class DTVD(nn.Module):
    def __init__(self, ndf):
        super(DTVD, self).__init__()
        self.slice1 = nn.Sequential(
            nn.Conv3d(3, ndf, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.slice2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, (4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.slice3 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ndf * 16),
            nn.LeakyReLU(inplace=True),
        )

        self.slice4 = nn.Sequential(
            nn.Conv3d(ndf * 16, 1, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
        )

    def forward(self, x):
        x1 = self.slice1(x)
        x2 = self.slice2(x1)
        x3 = self.slice3(x2)
        x4 = self.slice4(x3)
        return x4.view(-1, 1)

class DTVG(nn.Module):
    def __init__(self, ngf=64, dlatent_size=512, n_blocks=9, use_2d=False, use_flow=True):
        super(DTVG, self).__init__()
        self.use_flow = use_flow
        self.ngf = ngf
        self.dlatent_size = dlatent_size
        self.n_blocks = n_blocks
        self.use_2d = use_2d
        # shared encoder ------------------------------------- 1
        share = [nn.ReflectionPad2d(2),
                 nn.Conv2d(3, ngf, kernel_size=5, stride=1, padding=0, bias=False),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]
        share += [nn.Conv2d(ngf * 1, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(ngf * 2),
                  nn.ReLU(True)]
        share += [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(ngf * 2),
                  nn.ReLU(True)]
        self.share = nn.Sequential(*share)
        # img path ------------------------------------------- 2-1
        img_path = []
        for i in range(n_blocks):
            img_path += [ResnetBlock(ngf * 2, norm_layer=nn.BatchNorm2d, use_bias=False)]
        self.img_path = nn.Sequential(*img_path)
        # opt path ------------------------------------------- 2-2
        self.opt_path = G_synthesis(ngf, dlatent_size)
        # decoder -------------------------------------------- 3
        decoder = []
        ngf_out = 96
        if use_flow:
            decoder += [nn.Conv2d(ngf * 3, ngf_out, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf_out),
                        nn.ReLU(True)]
        else:
            decoder += [nn.Conv2d(ngf * 2, ngf_out, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(ngf_out),
                        nn.ReLU(True)]
        decoder += [nn.Conv2d(ngf_out, ngf_out, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(ngf_out),
                    nn.ReLU(True)]
        # using 2D
        if self.use_2d:
            decoder += [nn.ConvTranspose2d(ngf_out, ngf_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(ngf_out),
                        nn.ReLU(True)]
            decoder += [nn.ReflectionPad2d(2)]
            decoder += [nn.Conv2d(ngf_out, ngf_out, kernel_size=5, padding=0)]
            decoder += [nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)

        # decoder_3d ----------------------------------------- 4
        if not self.use_2d:
            decoder_3d = []
            decoder_3d += [nn.ConvTranspose3d(3, 3, 1, bias=False),
                           nn.BatchNorm3d(3),
                           nn.LeakyReLU(0.2, inplace=True)]
            decoder_3d += [nn.ConvTranspose3d(3, 3, 1, bias=False),
                           nn.BatchNorm3d(3),
                           nn.LeakyReLU(0.2, inplace=True)]
            decoder_3d += [nn.ConvTranspose3d(3, 3, (3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
                           nn.Tanh()]
            self.decoder_3d = nn.Sequential(*decoder_3d)
        

    def batch_norm_1d(self, x, gamma, beta):
        eps = 1e-5
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

    def resize_flow(self, flow, new_shape):
        _, _, h, w = flow.shape
        new_h, new_w = new_shape
        flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                               mode='bilinear', align_corners=True)
        scale_h, scale_w = h / float(new_h), w / float(new_w)
        flow[:, 0] /= scale_w
        flow[:, 1] /= scale_h
        return flow

    def forward(self, x, dlatent):  # [B, 3, 128, 128]
        # optical vector normalization
        gamma = torch.ones(dlatent.shape[1]).to(0)
        beta = torch.zeros(dlatent.shape[1]).to(0)
        dlatent = self.batch_norm_1d(dlatent, gamma, beta)
        # convs
        x = self.share(x)                           # [B, 128, 64, 64]
        x_img = self.img_path(x)                    # [B, 128, 64, 64]
        
        if self.use_flow:
            x_opt = self.opt_path(x, dlatent)           # [B, 32*2, 64, 64]
            x_cat = torch.cat([x_img, x_opt], dim=1)    # [B, 32*6, 64, 64]
            out = self.decoder(x_cat)  # [B, 32*2, 64, 64]
            out = out.view(out.shape[0], 3, 32, out.shape[2], out.shape[3])

            return out, x_opt
        else:
            out = self.decoder(x_img)
            out = out.view(out.shape[0], 3, 32, out.shape[2], out.shape[3])
            return out


if __name__ == '__main__':

    G = DTVG(ngf=64, dlatent_size=512, n_blocks=9, use_2d=True, use_flow=True)
    # net = torch.load('/home/lyc/ckp/videoprediction/mdgan_sky/200218142546/S1_G.pth')
    # G.load_state_dict(net)
    # D = VIPDFrame(3)
    # input = torch.randn(2, 3, 32, 128, 128)
    # output = D(input)
    # print(1)