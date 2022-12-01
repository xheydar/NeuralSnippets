import init

import sys
import os
import copy

import torch
import numpy as np
from torch import optim
from tqdm import tqdm

from datasets import datasets
#from noise_scheduler import NoiseScheduler
from diffusion_tools import DiffusionTools
from diffusion_transform import DiffusionTransform
from batch_generator import BatchGenerator

import model
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

        self._cfg = config( cfg_fname, tag )

        self.timesteps = self._cfg.params['timesteps']
        self.batch_size = self._cfg.params['batch_size']
        self.image_size = self._cfg.params['image_size']

        self.dataset = datasets['stanfordcars']('../data', image_size=self.image_size)

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

    def build_batches( self ):
        self.diffusion_tools = DiffusionTools(num_steps=self.timesteps, img_size=self.image_size)
        transform = DiffusionTransform( self.diffusion_tools )
        self.batches = BatchGenerator( self.dataset.dataset, self.batch_size, transform=transform, randomize=True )

    def load_model( self ):
        self.model = {}
        self.model['unet'] = model.UNet( num_classes=self.dataset.num_classes ).to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

    def do_stuff( self ):
        x_noisy, noise, t, y = self.batches[10]

        pred_noise = self.model['unet']( x_noisy, t, y )
        loss = self.model['loss']( pred_noise, noise.detach() )

        print( loss )

    def train( self, num_epoch=100 ):
        optimizer = optim.Adam(self.model['unet'].parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader( self.batches, batch_size=None, num_workers=8 )

        if not os.path.exists('./snapshots') : 
            os.makedirs('./snapshots')

        ema = EMA(0.995)
        ema_model = copy.deepcopy(self.model['unet']).eval().requires_grad_(False)

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

                pred_noise = self.model['unet']( x_noisy, t, y )
                loss = self.model['loss']( pred_noise, noise.detach() )

                loss_values.append( loss.item() )

                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, self.model['unet'])

            print( np.mean(loss_values) )

            if self._cfg.save_snapshots :
                state_dict = self.model['unet'].state_dict()
                torch.save({'state_dict':state_dict}, self._cfg.snapshots_tmp % (epoch+1))

        state_dict = self.model['unet'].state_dict()
        torch.save({'state_dict':state_dict}, self._cfg.model_path )

        state_dict = ema_model.state_dict()
        torch.save({'state_dict':state_dict}, self._cfg.ema_model_path )


if __name__=="__main__" :
    m = module('config.yaml',sys.argv[1])
    m.build_batches()
    m.load_model()
    m.train(num_epoch=300)

