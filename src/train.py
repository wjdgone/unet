import os
import torch
from torch.utils.data import DataLoader
# from tqdm import tqdm 
import matplotlib.pyplot as plt

from model import Model
from data import DSLoader
from visualizations import save_xyz



CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')

def train(opt, model_path=None): 
    ''' Train a model from scratch, or use a pre-trained model. '''

    # datasets
    train_ds = DataLoader(DSLoader(opt, 'train'), batch_size=opt['batch_size'], shuffle=True)
    val_ds = DataLoader(DSLoader(opt, 'val'), batch_size=1)

    # initialize model
    model = Model(opt)

    # load pre-trained model
    if model_path is not None: 
        model.load_model()

    # train epochs
    for epoch in range(1, opt['epochs']+1): 
        print(f"Epoch {epoch}/{opt['epochs']} start!")

        # train batches
        for idx, batch in enumerate(train_ds): 

            # get images and their labels
            train_imgs, train_lbls, _ = batch
            train_imgs, train_lbls = train_imgs.to(device), train_lbls.to(device)
        
            model.train_step(train_imgs, train_lbls)

        # save 5 outputs every 10 epochs
        if epoch % 10 == 0: 
            
            # val ds
            for idx, batch in enumerate(val_ds): 
                val_imgs, val_lbls, val_names = batch
                val_imgs, val_lbls = val_imgs.to(device), val_lbls.to(device)

                val_imgs, val_lbls, preds = model.test_step(val_imgs, val_lbls)
                
                # save visualization
                tag = f'val_epoch{epoch}_'
                save_xyz(val_imgs, val_lbls, preds, val_names, opt['training_results_dir'], tag)

                if idx > 4: 
                    break

    # save model
    model_path = os.path.join(opt['model_dir'], f"epoch_{opt['epochs']}.pt")
    torch.save(model.net.state_dict(), model_path)


if __name__ == '__main__': 


    from utils import TempDSLoader
    import yaml

    # import options
    OPT_PATH = './options.yaml'
    with open(OPT_PATH, 'r') as f: 
        opt = yaml.safe_load(f)

    train_ds = DataLoader(TempDSLoader(), batch_size=opt['batch_size'], shuffle=True)
    val_ds = DataLoader(TempDSLoader(), batch_size=1)

    train(train_ds, val_ds, opt)