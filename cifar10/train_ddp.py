import os
import torch 
import torch.nn as nn 

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

class DDPManager :
    def __init__( self ):
        self.world_size = int(os.environ.get("WORLD_SIZE",0))
        self.local_rank = int(os.environ.get("LOCAL_RANK",-1))

        if self.local_rank != -1 :
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self._device = self.local_rank

    def cleanup( self ):
        if self.local_rank != -1 and dist.is_initialized():
            dist.destroy_process_group()

    @property 
    def device( self ):
        return self._device





class Module :
    def __init__( self ):
        self.world_size = int(os.environ.get("WORLD_SIZE",0))
        self.local_rank = int(os.environ.get("LOCAL_RANK",-1))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.local_rank)
        self.device = self.local_rank

        print( self.world_size, self.local_rank )

        self.config = load_config('params.yaml')

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def load_model( self ):
        self.model = {}
        self.model['net'] = model.Net().to( self.device )
        self.model['loss'] = model.Loss().to( self.device )


        self.model['net'] = self.model['net'].to( self.device )
        self.model['loss'] = self.model['loss'].to( self.device )

        self.model['net'] = DDP(self.model['net'], device_ids=[self.device])

    def load_dataset( self ):
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

        self.datasets = {}
        self.datasets['train'] = dataset( self.config.dataset, transform, train=True )
        self.datasets['test'] = dataset( self.config.dataset, transform, train=False )

        self.datasets['sampler'] = DistributedSampler( self.datasets['train'] )

    def train( self ):

        sampler = self.datasets['sampler']
        model = self.model['net']
        loss = self.model['loss']

        optimizer = optim.Adam( model.parameters(), lr=1e-4)
        dataloader = DataLoader(self.datasets['train'],
                                batch_size=512, 
                                sampler=sampler, 
                                shuffle=False
                                )

        epochs = 100 
        for epoch in range(epochs):
            sampler.set_epoch(epoch)

            pbar = tqdm(dataloader, 
                    desc=f"Epoch {epoch+1}/{epochs}", 
                    disable=(self.local_rank != 0))


        
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
                optimizer.zero_grad()
                outputs = model(inputs)
                l = loss(outputs, labels)
            
                l.backward() 
                optimizer.step()

                if self.local_rank == 0:
                    pbar.set_postfix({'loss': f"{l.item():.4f}"})

        pass

        


if __name__=="__main__" :

    m = Module()


    try :
        m.load_model()
        m.load_dataset()

        m.train()
    finally :
        m.cleanup()
