import os 
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import Model
from data import DSLoader
from visualizations import save_xyz, save_z


CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')


def test(opt): 
    ''' Evaluate model on test dataset. '''

    # test dataset
    test_ds = DataLoader(DSLoader(opt, 'test'), batch_size=1)

    # choose model from last epoch
    model_name = os.listdir(opt['model_dir'])[-1]
    model_path = os.path.join(opt['model_dir'], model_name)
    
    # initialize model 
    model = Model(opt)

    # load model 
    model.load_model(model_path)

    # initialize metrics
    TP, FP, FN, TN = 0, 0, 0, 0

    # test each image
    for idx, batch in enumerate(test_ds): 

        # get images and their labels
        test_imgs, test_lbls, test_names = batch
        test_imgs, test_lbls = test_imgs.to(device), test_lbls.to(device)
    
        test_imgs, test_lbls, preds = model.test_step(test_imgs, test_lbls)

        # compute confusion matrix
        t = test_lbls.cpu().detach().numpy()
        p = preds.cpu().detach().numpy()

        t, p = np.reshape(t, (-1)), np.reshape(p, (-1))

        # print(t.shape, p.shape)
        tn, fp, fn, tp = confusion_matrix(t, p).ravel()

        TP += tp
        FP += fp
        FN += fn
        TN += tn

        # save visualizations
        # 1. image, label, and pred
        test_imgs = torch.unsqueeze(test_imgs, 0)
        test_lbls = torch.unsqueeze(test_lbls, 0)
        preds = torch.unsqueeze(preds, 0)
        save_xyz(test_imgs, test_lbls, preds, test_names, opt['testing_results_dir'])

        # 2. raw pred file
        save_z(preds, opt['testing_results_dir'], test_names)

    # calculate metrics 
    acc = (TP + TN) / (TP + FP + FN + TN)
    pre = TP / (TP + FP)
    rcl = TP / (TP + FN)
    f1s = (2*pre*rcl) / (pre + rcl)

    print('accuracy - ', round(acc, 4))
    print('precision - ', round(pre, 4))
    print('recall - ', round(rcl, 4))
    print('f1-score - ', round(f1s, 4))