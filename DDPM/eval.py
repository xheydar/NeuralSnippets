import init

import os

import torch
import numpy as np
from torch import optim
from tqdm import tqdm

from datasets import datasets
from diffusion_tools import DiffusionTools
from diffusion_transform import DiffusionTransform
from batch_generator import BatchGenerator

from models import models
from transforms import transforms
from config import config

from matplotlib import pyplot as pp
pp.ion()

class module :
    def __init__( self, cfg_fname, tag ):
        self.cfg = config( cfg_fname, tag )

        self.timesteps = self.cfg.params['timesteps']
        self.batch_size = self.cfg.params['batch_size']
        self.image_size = self.cfg.params['image_size']

        self.dataset = datasets[self.cfg.dataset]('../data', image_size=self.image_size)

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.diffusion_tools = DiffusionTools(num_steps=self.timesteps, img_size=self.image_size)

    def load_model( self ):
        self.model = {}
        self.model['net'] = models[self.cfg.model]( num_classes=self.dataset.num_classes ).to(self.device)
        self.model['loss'] = models['loss'].to(self.device)

        model_data = torch.load( self.cfg.ema_model_path, map_location='cpu' )
        self.model['net'].load_state_dict( model_data['state_dict'], strict=True )

    def calculate( self ):
        labels = np.array([2], dtype=np.int64 )
        labels = torch.from_numpy(labels)
        self.out = self.diffusion_tools.sample( self.model['unet'], 1, labels )

    def do_stuff( self ):
        self.diffusion_tools.sample( self.model['net'], 1, None )

        
        
