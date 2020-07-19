import os
from abc import abstractmethod, ABCMeta
import torch
import torchvision
from torch.utils.data import Dataset
from datasets.utils_dataset import is_npy_file, default_loader
import numpy as np
from torchvision import transforms
torchvision.set_image_backend('accimage')

class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root):
        self.root = root

    def _collect_samples(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pass


class SkyTimelapse(BaseDataset):
    def __init__(self, root, split='sky_train', type='', nframes=32, transform=None, target_transform=None, loader=default_loader):
        super(SkyTimelapse, self).__init__('{}/{}'.format(root, split))
        self.root = root
        self.split = split
        self.nframes = nframes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.type = type

        self._collect_samples_Flow()
        # transform
        self.transformflow = transforms.Compose([transforms.ToTensor()])
        self.transformimg = transforms.Compose([transforms.Resize((128, 128)),
                                                transforms.ToTensor(),])

    def _collect_samples_Flow(self):
    
        self.samples = []
        classes_former = ''
        self.root_flow = os.path.join(self.root, 'flow', self.split)
        self.root = os.path.join(self.root, self.split)
        for subfolder in os.listdir(self.root_flow):
            subpath = os.path.join(self.root_flow, subfolder)
            if not os.path.isdir(subpath):
                continue
            for classes in os.listdir(subpath):
                flowfolder = os.path.join(self.root_flow, subfolder, classes)
                imgfolder = os.path.join(self.root, subfolder, classes)

                if os.path.isdir(flowfolder):
                    nsubflow = sorted(os.listdir(flowfolder))

                    if len(nsubflow) < 32:
                        continue
                    i = 0
                    item_flow = []
                    item_frames = []

                    for fi in sorted(nsubflow):
                        if is_npy_file(fi):
                            i = i + 1
                            flow_name = fi
                            img_name = fi.split('.npy')[0] + '.jpg'

                            flow_path = os.path.join(flowfolder, flow_name)
                            img_path = os.path.join(imgfolder, img_name)

                            item_flow.append(flow_path)
                            item_frames.append(img_path)

                            if i % self.nframes == 0:
                                first_img_path = sorted(item_frames)[0]       
                                self.samples.append([item_flow, item_frames, first_img_path, classes])
                                break


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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ## origin
        clip_flow, clip_frame, FirstFrame, cls = self.samples[idx]
        img_clip = []
        flow_clip = []
        flow_gt_clip = []

        for i, path in enumerate(clip_frame):
            img = self.loader(path)
            img = self.transformimg(img)
            flow = np.load(clip_flow[i])
            flow = self.transformflow(flow)

            flow_gt = self.resize_flow(flow, (64, 64))

            flow = flow.view(flow.size(0), 1, flow.size(1), flow.size(2))
            flow_clip.append(flow)
            img = img.view(img.size(0), 1, img.size(1), img.size(2))  # reshape to 3 *1 *64*64
            img_clip.append(img)

            flow_gt_clip.append(flow_gt)

        img_first = self.loader(FirstFrame)
        img_first = self.transform(img_first)

        img_frames = torch.cat(img_clip, 1)  # size 3*32*64*64
        img_flows = torch.cat(flow_clip, 1)
        img_gt_flows = torch.cat(flow_gt_clip, 0)

        return img_frames, img_flows, img_gt_flows, img_first, cls

if __name__ == '__main__':
    dataset = SkyTimelapse('/home/lyc/code/release/vpgan_init/data/sky_timelapse/', split='sky_train')
    dataset.__getitem__(0)
    print(len(dataset))
    #test('/home/lyc/data/motion_prediction/sky_timelapse/sky_train')
