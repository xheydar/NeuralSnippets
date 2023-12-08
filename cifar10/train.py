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

cfg = edict()
cfg.dataset = edict();

if platform.system() == "Darwin":
    cfg.dataset.root = '/Users/heydar/Work/void/cache'
else :
    cfg.dataset.root = '/home/heydar/cache'

class train(trainer) :
    def __init__( self ):
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
    parser.add_argument('-t','--tag',help="Experiment tag", required=True)
    parser.add_argument('-n','--nepoch',help="Number of epoches", default=100, type=int)
    parser.add_argument('--ema', help="User EMA", action='store_true', default=False )
    args = parser.parse_args()
    return args
    

if __name__=="__main__" :

    args = parse_commandline()

    #params = tools.yaml_loader('params.yaml')
    #print( params )

    a = api('http://10.8.1.2:3000/api/logs/update-experiment/',
            'L8FX1wg70BEzzZQ9UtiJ')

    t = train()
    t.load_dataset()
    t.load_model()
    data = t.train( nepoch=int(args.nepoch), use_ema=args.ema, api=a )

    save_path = f'results/{args.tag}.pkl'

    with open(save_path,'wb') as ff :
        pickle.dump( data, ff )

    print( "Saved to %s" % (save_path) )
