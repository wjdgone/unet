import os
import torch 
from torch.utils.data import Dataset 


def make_subfolder_idf(opt, *keys): 
    ''' Designate a subfolder with the name 'idf'. Note that opt is returned. '''

    for key in keys: 
        new_key = key.replace('_root', '')
        opt[new_key] = os.path.join(new_key, opt['idf'])

    return opt


def make_dir(dir): 
    ''' Make directory if it doesn't exist. '''

    try: 
        os.makedirs(dir)
        print(f'folder created at {dir}')
    except FileExistsError: 
        pass


def make_dirs_auto(*dirs): 
    ''' Makes directories passed. '''

    for dir in dirs: 
        make_dir(dir)


# for debugging 
class TempDSLoader(torch.utils.data.Dataset): 
    def __init__(self): 

        # img_size = 572
        img_size = 256
        self.imgs = torch.randint(low=0, high=255,size=(24,3,img_size,img_size))
        self.lbls = torch.randint(low=0, high=1,size=(24,1,img_size,img_size), dtype=torch.uint8)

    def __len__(self): 
        return self.imgs.size()[0]


    def __getitem__(self, idx): 
        return self.imgs[idx], self.lbls[idx]


if __name__ == '__main__': 

    ds = TempDSLoader()
    print('num images in dataloader: ', len(ds))
    imgs, lbls = ds[0]
    print('size of 0th image in dataloader: ', imgs.size())