import os
import torch 
import torch.nn as nn 
import time

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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def load_config( cfg_path ):

    with open(cfg_path,'r') as ff :
        cfg = yaml.safe_load(ff)
    
    return munchify(cfg)

class ResourceMananger :
    def _select_device( self, device=None ):
        if isinstance(device, torch.device): 
            return device

        if device is not None :
            return torch.device(device)

        if torch.cuda.is_available() and device is None :
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and device is None :
            return torch.device("mps")
        return torch.device("cpu")
    def __init__( self, device=None ):


        if self.is_distributed :
            self.world_size = int(os.environ.get("WORLD_SIZE",0))
            self.local_rank = int(os.environ.get("LOCAL_RANK",-1))
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self._device = self.local_rank
        else :
            self._device = self._select_device( device )

    def cleanup( self ):
        if self.is_distributed and dist.is_initialized() :
            dist.destroy_process_group()

    @property 
    def is_distributed( self ):
        return "WORLD_SIZE" in os.environ

    @property 
    def device( self ):
        return self._device

    @property
    def is_master( self ):
        if not self.is_distributed :
            return True 
        return self.local_rank == 0 


class Module :
    def __init__( self ):
        self.resource = ResourceMananger()
        self.config = load_config('params.yaml')

    def cleanup(self):
        self.resource.cleanup() 

    def load_model( self ):
        self.model = {}
        self.model['net'] = model.Net().to( self.resource.device )
        self.model['loss'] = model.Loss().to( self.resource.device )

        if self.resource.is_distributed :
            self.model['net'] = DDP(self.model['net'], device_ids=[self.resource.device])

    def load_dataset( self ):
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

        self.datasets = {}
        self.datasets['train'] = dataset( self.config.dataset, transform, train=True )
        self.datasets['test'] = dataset( self.config.dataset, transform, train=False )

        if self.resource.is_distributed :
            self.datasets['sampler'] = DistributedSampler( self.datasets['train'] )

    def train( self ):

        sampler = self.datasets.get('sampler',None)
        model = self.model['net']
        loss = self.model['loss']

        optimizer = optim.Adam( model.parameters(), lr=1e-4)
        dataloader = DataLoader(self.datasets['train'],
                                batch_size=1024, 
                                sampler=sampler, 
                                shuffle=not self.resource.is_distributed
                                )

        t0 = time.time()

        epochs = 100 
        for epoch in range(epochs):

            if sampler is not None :
                sampler.set_epoch(epoch)

            pbar = tqdm(dataloader, 
                    desc=f"Epoch {epoch+1}/{epochs}", 
                    disable=not self.resource.is_master
            )

 
            for inputs, labels in pbar:
                inputs = inputs.to(self.resource.device)
                labels = labels.to(self.resource.device)
            
                optimizer.zero_grad()
                outputs = model(inputs)
                l = loss(outputs, labels)
            
                l.backward() 
                optimizer.step()

                if self.resource.is_master:
                    pbar.set_postfix({'loss': f"{l.item():.4f}"})

        t1 = time.time()

        print(f"Elapsed time : {t1-t0}")
        


if __name__=="__main__" :

    m = Module()


    try :
        m.load_model()
        m.load_dataset()

        m.train()
    finally :
        m.cleanup()
