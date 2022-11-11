import init

import sys
import numpy as np

import torch
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from datasets import datasets
import model

from matplotlib import pyplot as pp
pp.ion()

class module :
    def __init__( self, dset ):
        self.dset = dset

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])


        self.dataset = datasets[self.dset](root='../data', train=True, transform=transform)
        #self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset.dataset, batch_size=128, shuffle=True)

    def build_model( self ):
        indim = self.dataset.dim
        self.latent_size = 10

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['generator'] = model.Generator( self.latent_size, self.dataset.shape ).to(self.device)
        self.model['discriminator'] = model.Discriminator( self.dataset.shape).to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

        model_data = torch.load( 'model_%s.pt' % ( self.dset ) , map_location='cpu' )

        self.model['generator'].load_state_dict(  model_data['state_dict']['generator'], strict = True )
        self.model['discriminator'].load_state_dict( model_data['state_dict']['discriminator'], strict = True )

    def do_stuff( self ):
        z = torch.randn(64, self.latent_size)
        samples = self.model['generator'](z)
        samples = samples.reshape([-1,1,28,28]).detach().cpu().numpy()
 
        pp.matshow(samples[1][0])

        pass

