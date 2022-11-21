import init

import os

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

from matplotlib import pyplot as pp
pp.ion()

class module :
    def __init__( self ):

        self.timesteps = 300
        self.batch_size = 64
        self.image_size = 64

        self.dataset = datasets['cifar10']('../data', image_size=self.image_size)

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

    def build_batches( self ):

        self.diffusion_tools = DiffusionTools(num_steps=self.timesteps, img_size=self.image_size)
        transform = DiffusionTransform( self.diffusion_tools )
        self.batches = BatchGenerator( self.dataset.dataset, self.batch_size, transform=transform, randomize=True )

    def load_model( self ):
        self.model = {}
        self.model['unet'] = model.UNet().to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

    def do_stuff( self ):
        x_noisy, noise, t, y = self.batches[10]
        
        pred_noise = self.model['unet']( x_noisy, t )
        loss = self.model['loss']( pred_noise, noise.detach() )

        print( loss )

    def train( self, num_epoch=100 ):
        optimizer = optim.Adam(self.model['unet'].parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader( self.batches, batch_size=None, num_workers=8 )

        if not os.path.exists('./snapshots') : 
            os.makedirs('./snapshots')

        for epoch in range(num_epoch):
            print("Epoch : %d/%d" % ( epoch+1, num_epoch ) )
            loss_values = []
            for batch_idx, ( x_noisy, noise, t, y ) in tqdm(enumerate(dataloader)):
                x_noisy = x_noisy.to(self.device)
                noise = noise.to(self.device)
                t = t.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                pred_noise = self.model['unet']( x_noisy, t )
                loss = self.model['loss']( pred_noise, noise.detach() )

                loss_values.append( loss.item() )

                loss.backward()
                optimizer.step()

            print( np.mean(loss_values) )
            state_dict = self.model['unet'].state_dict()
            torch.save({'state_dict':state_dict}, './snapshots/model_snapshot_%d.pt' % (epoch+1))

if __name__=="__main__" :

    m = module()
    m.build_batches()
    m.load_model()
    m.train(num_epoch=300)





