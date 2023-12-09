import argparse
import pickle

import torch 
import torchvision
import numpy as np
import platform
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import yaml

from easydict import EasyDict as edict

from dataset import dataset
import model

from trainer import trainer
import tools

from api import api
from validate import validate

cfg = edict()
cfg.dataset = edict();

if platform.system() == "Darwin":
    cfg.dataset.root = '/Users/heydar/Work/void/cache'
else :
    cfg.dataset.root = '/home/heydar/cache'

class train(trainer) :
    def __init__( self, params ):
        super().__init__( params )

        if torch.cuda.is_available() :
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() :
            self.device = torch.device("mps")
        else :
            self.device = torch.device("cpu")

    def load_dataset( self ):

        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

        self.datasets = {}
        self.datasets['train'] = dataset( cfg.dataset, transform, train=True )
        self.datasets['test'] = dataset( cfg.dataset, transform, train=False )

    def load_model( self ):
        self.model = {}
        self.model['net'] = model.Net().to( self.device )
        self.model['loss'] = model.Loss().to( self.device )

    def do_stuff( self ):

        inputs = []
        labels = []

        for i in range(16):
            t,l = self.datasets['train'][i]

            t = t.unsqueeze(0)

            inputs.append(t)
            labels.append(l)

        inputs = torch.cat( inputs, dim=0 ).to( self.device )
        labels = torch.from_numpy( np.array(labels,dtype=np.int64 ) ).to( self.device )

        outputs = self.model['net']( inputs ) 
        loss = self.model['loss']( outputs, labels )

        print( outputs.shape )
        print( loss )

        #print( outputs.shape )
        
def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', help="API Key", default=None)
    args = parser.parse_args()
    return args
    
if __name__=="__main__" :
    args = parse_commandline()
    params = tools.yaml_loader('params.yaml')

    if args.api :
        a = api('%s/api/logs/update-experiment/' % (params['api']['server']), args.api)
    else :
        a = None

    t = train( params )
    val = validate()
    t.load_dataset()
    t.load_model()
    t.train( api=a, validate=val )

