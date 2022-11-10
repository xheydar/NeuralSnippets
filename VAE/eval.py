import init

import torch
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

import model

from matplotlib import pyplot as pp
pp.ion()

class module :
    def __init__( self ):
        pass
        #self.dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
        #self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)

    def build_model( self ):
        latent_size = 2
        layers = [784, 512, 256]

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['network'] = model.Net( layers, latent_size ).to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

        model_data = torch.load( 'model.pt' , map_location='cpu' )
        self.model['network'].load_state_dict(  model_data['state_dict'], strict = True )

    def do_stuff( self ):    
        z = torch.randn(64, 2)
        samples = self.model['network'].decoder(z)

        samples = samples.reshape([-1,1,28,28]).detach().cpu().numpy()
 
        pp.matshow(samples[1][0])

