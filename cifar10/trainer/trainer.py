import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from torch.optim import lr_scheduler

from .ema import ModelEMA
from config import Params
from notes import api

class trainer :
    def load_config( self ):
        self.params = Params( self._cfg_name )

        if self.params.api and self._api_key :
            self.api = api( '%s/api/logs/update-experiment/' % self.params.api['server'], 
                           self._api_key )
        else :
            self.api = None

    def __init__( self, cfg_name, api_key=None ):
        self._api_key = api_key
        self._cfg_name = cfg_name 

        self.load_config()

        #self.params = params
        #self.loss = loss
        #self.validate = validate
        #self.api = api

    def get_optimizer( self, name='SGD', lr=0.001, momentum=0.9, decay=1e-5 ): 
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.model['net'].modules():
            for p_name, p in v.named_parameters(recurse=0):
                if p_name == 'bias':  # bias (no decay)
                    g[2].append(p)
                elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                    g[1].append(p)
                else:
                    g[0].append(p)  # weight (with decay)


        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        #print(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        #      f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
        return optimizer

    def warmup_step( self, optimizer, ni, epoch_idx ):
        xi = [0, self.nw]
    
        self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs/self.batch_size]).round())

        for j,x in enumerate( optimizer.param_groups ):
            x['lr'] = np.interp( ni, xi, [ self.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch_idx) ] )
            if 'momentum' in x :
                x['momentum'] = np.interp(ni,xi,[ self.warmup_momentum, self.optimizer_momentum ])

    def train_step( self, epoch_idx, train_loader, optimizer, ema=None, use_amp=False ):
        self.model['net'].train()

        mloss = 0.0

        optimizer.zero_grad()

        for idx, data in tqdm(enumerate( train_loader, 0 )):
            ni = idx + epoch_idx * self.nb

            if ni < self.nw :
                self.warmup_step( optimizer, ni, epoch_idx )
    
            loss = self.compute_loss( self.model, data, self.device, use_amp=use_amp )
            loss.backward()

            if ni - self.last_opt_step > self.accumulate :
                torch.nn.utils.clip_grad_norm_( self.model['net'].parameters(), max_norm=10.0 )

                optimizer.step()
                optimizer.zero_grad()

                if ema :
                    ema.update( self.model['net'] )

                self.last_opt_step = ni

            mloss = ( mloss * idx + float(loss) ) / (idx + 1)

        return mloss

    def train( self ):
        nepoch = self.params.trainer['nepoch']
        batch_size = self.params.trainer['batch_size']
        accumulate_batch_size = self.params.trainer['accumulate_batch_size']
        eval_batch_size_multiplier = self.params.trainer['eval_batch_size_multiplier']
        use_ema = self.params.trainer['use_ema']
        use_amp = self.params.trainer['use_amp']
        lrf = self.params.trainer['lrf']

        # Warmup 
        warmup_nepoch = self.params.trainer['warmup_nepoch']
        warmup_bias_lr = self.params.trainer['warmup_bias_lr']
        warmup_momentum = self.params.trainer['warmup_momentum']

        optimizer_momentum = self.params.trainer['optimizer']['momentum']

        train_loader = self.datasets['train'].get_loader( batch_size, True )
        test_loader = self.datasets['test'].get_loader( batch_size * eval_batch_size_multiplier , False )

        optimizer = self.get_optimizer( **self.params.trainer['optimizer'] )

        if use_ema :
            ema = ModelEMA( self.model['net'] )
        else :
            ema = None

        print( ['train_loss'] + self.validate.data_keys if self.validate else ['train_loss'] )

        if self.api :
            self.api.reset()
            self.api.add_cfg('nepoch', nepoch)
            self.api.add_cfg('keys', ['train_loss'] + self.validate.data_keys if self.validate else ['train_loss'])
            self.api.send('started')

        self.ni = 0 
        self.nb = len(train_loader) 
        self.nbs = accumulate_batch_size
        self.batch_size = batch_size 
        self.accumulate = accumulate_batch_size / batch_size
        self.last_opt_step = -1

        self.nw = warmup_nepoch * self.nb #max(round(warmup_nepoch * self.nb),100)
        self.warmup_bias_lr = warmup_bias_lr 
        self.warmup_momentum = warmup_momentum
        self.optimizer_momentum = optimizer_momentum

        self.lf = lambda x: (1 - x / nepoch) * (1.0 - lrf) + lrf  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf) 

        assert hasattr(self,'compute_loss'), "compute_loss was not provided."
        if not hasattr(self,'validate'):
            self.validate = None

        for epoch in range( nepoch ):

            print(f'Epoch {epoch+1}/{nepoch}')

            ave_loss = self.train_step( epoch, train_loader, optimizer, ema, use_amp=use_amp )

            epoch_data = {'train_loss':ave_loss, 'epoch':epoch}

            if self.validate :
                val_data = self.validate( test_loader, ema.ema if ema else self.model['net'], self.device  )
                epoch_data.update( val_data )

            scheduler.step()

            if self.api :
                self.api.add_item(epoch_data)
                self.api.send('running' if epoch < nepoch-1 else 'done' )
