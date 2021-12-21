import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 


def save_xyz(imgs, lbls, preds, names, save_dir, tag=None): 
    ''' Save a figure of the image, label, and prediction. 
        - imgs, lbls, preds shape : (batch size, channels, y, x) '''

    for i in range(imgs.size()[0]): 

        img = imgs[i].cpu().detach().numpy()
        lbl = lbls[i].cpu().detach().numpy()
        pred = preds[i].cpu().detach().numpy()

        img = np.uint8(np.transpose(img, (1,2,0)))
        lbl = np.transpose(lbl, (1,2,0))
        pred = np.transpose(pred, (1,2,0))

        name = names[i]

        plt.close()
        plt.figure(figsize=(10,8))
        plt.subplot(1,3,1), plt.imshow(img)
        plt.subplot(1,3,2), plt.imshow(lbl)
        plt.subplot(1,3,3), plt.imshow(pred)
        
        if tag is not None: 
            save_path = os.path.join(save_dir, tag+name)
        else: 
            save_path = os.path.join(save_dir, name)
        plt.savefig(save_path)


def save_z(preds, save_dir, names): 
    ''' Save the prediction. '''

    for i in range(preds.size()[0]): 

        pred = preds[i].cpu()
        pred = torch.squeeze(pred)
        pred = torch.where(pred==0, 0, 255).to(torch.uint8)
        pred = torch.unsqueeze(pred, -1)
        pred = pred.detach().numpy()

        name = names[i]

        save_path = os.path.join(save_dir, name[:-4] + '.png')
        cv2.imwrite(save_path, pred)