import os
import torch 
import torch.nn as nn 

import argparse
import pickle

import torchvision
import numpy as np
import platform

from munch import munchify


import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import yaml


from dataset import dataset
import model


def load_config( cfg_path ):

    with open(cfg_path,'r') as ff :
        cfg = yaml.safe_load(ff)
    
    return munchify(cfg)



class Module :
    def __init__( self ):
        self.world_size = int(os.environ.get("WORLD_SIZE",0))
        self.local_rank = int(os.environ.get("LOCAL_RANK",-1))

        self.config = load_config('params.yaml')


if __name__=="__main__" :
    m = Module()

