import init

import sys
import os
import copy

import torch
from torch import nn
import numpy as np
from torch import optim
from tqdm import tqdm

from datasets import datasets
#from noise_scheduler import NoiseScheduler
from diffusion_tools import DiffusionTools
from batch_generator import BatchGenerator

from models import models
from transforms import transforms
from config import config

from matplotlib import pyplot as pp
pp.ion()


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

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

    def dev_test( self ):
        print( self.dataset.dataset[0] )


    def build_batches( self ):
        self.diffusion_tools = DiffusionTools(num_steps=self.timesteps, img_size=self.image_size)
        transform = transforms[self.cfg.transform]( self.diffusion_tools )
        self.batches = BatchGenerator( self.dataset.dataset, self.batch_size, transform=transform, randomize=True )

    def load_model( self ):
        self.model = {}
        self.model['net'] = models[self.cfg.model]( num_classes=self.dataset.num_classes ).to(self.device)
        self.model['loss'] = models['loss']().to(self.device)

        self.model['net'] = nn.DataParallel(self.model['net'])

    def do_stuff( self ):
        x_noisy, noise, t, y = self.batches[10]

        pred_noise = self.model['net']( x_noisy, t, y )
        loss = self.model['loss']( pred_noise, noise.detach() )

    def save_model( self, model, path ):
        try :
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()

        torch.save({'state_dict':state_dict}, path )
        
    def train( self, num_epoch=100 ):
        optimizer = optim.Adam(self.model['net'].parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader( self.batches, batch_size=None, num_workers=8 )

        if not os.path.exists('./snapshots') : 
            os.makedirs('./snapshots')

        ema = EMA(0.995)
        ema_model = copy.deepcopy(self.model['net']).eval().requires_grad_(False)

        for epoch in range(num_epoch):
            print("Epoch : %d/%d" % ( epoch+1, num_epoch ) )
            loss_values = []
            for batch_idx, ( x_noisy, noise, t, y ) in tqdm(enumerate(dataloader)):
                x_noisy = x_noisy.to(self.device)
                noise = noise.to(self.device)
                t = t.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                if np.random.random() < 0.1:
                    y = None

                pred_noise = self.model['net']( x_noisy, t, y )
                loss = self.model['loss']( pred_noise, noise.detach() )

                loss_values.append( loss.item() )

                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, self.model['net'])

            print( np.mean(loss_values) )

            if self.cfg.save_snapshots :
                self.save_model( self.model['net'], self.cfg.snapshots_tmp % (epoch+1))

        self.save_model( self.model['net'], self.cfg.model_path )
        self.save_model( ema_model, self.cfg.ema_model_path )

if __name__=="__main__" :
    m = module(sys.argv[1],sys.argv[2])
    m.build_batches()
    m.load_model()
    m.train(num_epoch=300)

