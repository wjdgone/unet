import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt 


class DSLoader(Dataset): 
    ''' Dataloader for custom dataset. '''

    def __init__(self, opt, mode): 
        
        self.opt = opt
        self.data_dir = os.path.join(opt['data_dir'], mode)
        self.names = os.listdir(os.path.join(self.data_dir, 'imgs'))


    def __len__(self): 
        return len(self.names)


    def __getitem__(self, idx): 
        
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.data_dir, 'imgs', name))
        lbl = cv2.imread(os.path.join(self.data_dir, 'lbls', name))
        
        # resize and pad image
        img = self.resize_image(img)
        lbl = self.resize_image(lbl)

        # (channels, y, x)
        img = np.moveaxis(img, -1, 0)
        lbl = np.moveaxis(lbl, -1, 0)

        # make binary mask
        lbl = np.where(lbl==0, 1, 0)
        lbl = lbl[:1, ...]

        # convert to torch
        img = torch.from_numpy(img)
        lbl = torch.from_numpy(lbl)

        return img, lbl, name


    def resize_image(self, img): 
        ''' img: numpy array in (y, x, band num) format. '''

        band_num = img.shape[-1]

        # numbers from unet paper
        img_size = 388
        tot_size = 572
        pad_size = (tot_size - img_size)//2
        img_resize = np.zeros((572, 572, band_num))

        # square images/labels
        img = cv2.resize(img, (img_size, img_size))

        # pad images/labelsx
        for band in range(band_num): 
            channel = img[..., band]
            channel = np.pad(channel, pad_size, 'reflect')
            img_resize[..., band] = channel

        return img_resize