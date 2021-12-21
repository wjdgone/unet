import time
import sys
import yaml

from utils import make_subfolder_idf, make_dirs_auto
from train import train
from test import test


OPT_PATH = './options.yaml'


def main(mode, opt): 
    ''' Prepare dataset for training and/or testing. '''

    if 'train' in mode: 
        train(opt, model_path=None)

    if 'test' in mode: 
        test(opt)


if __name__ == '__main__': 

    start_time = time.time()

    mode = sys.argv[1] # train, test, or traintest

    # import options
    with open(OPT_PATH, 'r') as f: 
        opt = yaml.safe_load(f)

    # parse directories
    opt = make_subfolder_idf(opt, 'model_root_dir', 
                             'training_results_root_dir',
                             'testing_results_root_dir',
                             )

    # create directories
    make_dirs_auto(opt['data_dir'], 
                   opt['model_dir'],
                   opt['training_results_dir'], 
                   opt['testing_results_dir'],
                   )

    main(mode, opt) 

    print(f'Finished. Time elapsed: {(time.time() - start_time)/60:.4f} mins.')