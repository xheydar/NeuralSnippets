import init

import torch
import numpy as np
from torch import optim
from tqdm import tqdm

from datasets import datasets
from noise_scheduler import NoiseScheduler
from batch_generator import BatchGenerator

import model

from matplotlib import pyplot as pp
pp.ion()

class module :
    def __init__( self ):

        self.timesteps = 300
        self.batch_size = 64

        self.dataset = datasets['stanfordcars']('../data', image_size=64)
        self.noise_scheduler = NoiseScheduler( self.timesteps, start=0.0001, end=0.02 )

    def build_batches( self ):
        self.batches = BatchGenerator( self.dataset.dataset, self.batch_size, randomize=True )

    def load_model( self ):
        
        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['unet'] = model.SimpleUnet().to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

    def do_stuff( self ):
        x = self.batches[10].to(self.device)

        t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device).long()

        x_noisy, noise = self.noise_scheduler.forward_diffusion_sample( x, t )
        noise_pred = self.model['unet'](x_noisy, t)
        loss = self.model['loss']( noise, noise_pred )

        print( loss )

        print( loss.item() )

        #t = torch.Tensor([50]).type(torch.int64)
        #image, noise = self.noise_scheduler.forward_diffusion_sample( batch, t )

        #image = self.dataset.reverse_transform( image[0] )
        #pp.imshow(image)

    def train( self, num_epoch=100 ):
        optimizer = optim.Adam(self.model['unet'].parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader(dataset=self.dataset.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(num_epoch):
            loss_values = []
            for batch_idx, batch in tqdm(enumerate(dataloader)):

                optimizer.zero_grad()

                x = batch[0].to(self.device)
                t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device).long()
                
                x_noisy, noise = self.noise_scheduler.forward_diffusion_sample( x, t )
                noise_pred = self.model['unet'](x_noisy, t)
                loss = self.model['loss']( noise, noise_pred )

                loss_values.append( loss.item() )

                loss.backward()
                optimizer.step()
            print( np.mean(loss_values) )

if __name__=="__main__" :

    m = module()
    m.load_model()
    m.train()





