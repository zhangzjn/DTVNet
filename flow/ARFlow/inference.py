import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite
import os

import warnings
warnings.filterwarnings("ignore")

class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/Sky/sky_ckpt.pth.tar')
    parser.add_argument('--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('--save_shape', default=[128, 128], type=int, nargs=2)
    parser.add_argument('--img_list', nargs='+', default=['examples/cloud_1.jpg', 'examples/cloud_2.jpg'])
    parser.add_argument('--root_split', default='sky_train')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--root', default='../../data/sky_timelapse/')
    parser.add_argument('--save_path', default='../../data/sky_timelapse/flow')
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    ts = TestHelper(cfg)
    if args.show:
        imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
        h, w = imgs[0].shape[:2]
    
        flow_12 = ts.run(imgs)['flows_fw'][0]
    
        flow_12 = resize_flow(flow_12, (h, w))
        np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
    
        vis_flow = flow_to_image(np_flow_12)
    
        fig = plt.figure()
        plt.imshow(vis_flow)
        plt.show()
    else:
        root = args.root + '/{}'.format(args.root_split)
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    
        samples = []
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_path = args.save_path + '/{}'.format(args.root_split)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
        for idx_c, cls in enumerate(classes):
    
            subfolder_path = os.path.join(root, cls)
            save_subfolder_path = os.path.join(save_path, cls)
    
            if not os.path.exists(save_subfolder_path):
                os.mkdir(save_subfolder_path)
    
            for idx_sc, subsubfold in enumerate(sorted(os.listdir(subfolder_path))):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold)):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                    save_subsubfolder_path = os.path.join(save_subfolder_path, subsubfold)
    
                    if not os.path.exists(save_subsubfolder_path):
                        os.mkdir(save_subsubfolder_path)
    
                    files = sorted(os.listdir(subsubfolder_path))
                    j = 0
                    count = 0
                    for i in range(0,(len(files) - 1),1):
    
                        if i  == 32 * j:
                            file_name1 = files[i]
                            file_name2 = files[i]
                        else:
                            file_name1 = files[i - 1]
                            file_name2 = files[i]
    
                        npy_file = os.path.join(save_subsubfolder_path, '{}.npy'.format(file_name2.split('.jpg')[0]))
    
                        file_path1 = os.path.join(subsubfolder_path, file_name1)
                        file_path2 = os.path.join(subsubfolder_path, file_name2)
                        s = {'imgs': [file_path1, file_path2]}
    
                        imgs = [imageio.imread(img).astype(np.float32) for img in s['imgs']]
                        h, w = imgs[0].shape[:2]
    
                        flow_12 = ts.run(imgs)['flows_fw'][0]
                        flow_12 = resize_flow(flow_12, (h, w))
                        flow_12_resize = resize_flow(flow_12, (128, 128))
    
                        np_flow_12_resize = flow_12_resize[0].detach().cpu().numpy().transpose([1, 2, 0])
                        np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
                        np.save(npy_file, np_flow_12_resize)
    
    
                        print('\rsub_cls_file: {}/{}'.format(i, len(files) - 1), end='')
    
                    print('\nsub_cls {}/{}'.format(idx_sc, len(os.listdir(subfolder_path))))
            print('cls {}/{}'.format(idx_c, len(classes)))
