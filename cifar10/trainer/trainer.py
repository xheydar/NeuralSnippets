import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from .ema import ModelEMA

class trainer :
    def __init__( self, params ):
        self.params = params

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

    def train_step( self, epoch_idx, train_loader, optimizer, ema=None ):

        self.model['net'].train()

        mloss = 0.0

        optimizer.zero_grad()

        for idx, data in tqdm(enumerate( train_loader, 0 )):
            self.ni = idx + epoch_idx * self.nb


            inputs, labels = data 

            inputs = inputs.to( self.device )
            labels = labels.to( self.device )

            #optimizer.zero_grad()

            outputs = self.model['net']( inputs )

            loss = self.model['loss']( outputs, labels )

            loss.backward()

            if self.ni - self.last_opt_step > self.accumulate :
                torch.nn.utils.clip_grad_norm_( self.model['net'].parameters(), max_norm=10.0 )
                optimizer.step()
                optimizer.zero_grad()

                if ema :
                    ema.update( self.model['net'] )

                self.last_opt_step = self.ni


            mloss = ( mloss * idx + float(loss) ) / (idx + 1)

        return mloss

    def eval_step( self, test_loader, ema=None ):

        if ema :
            model = ema.ema 
        else :
            model = self.model['net']

        model.eval()

        corrects = 0;
        total = 0

        for idx, data in tqdm(enumerate(test_loader), 0):
            inputs, labels = data 

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            pred = self.model['net']( inputs )
            pred = torch.argmax( pred, dim=1 )

            pred = pred.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            inds = np.where( pred == labels )[0]

            corrects += len(inds)
            total += len(labels)

        return corrects / total

    def train( self, api=None ):
        nepoch = self.params['trainer']['nepoch']
        batch_size = self.params['trainer']['batch_size']
        accumulate_batch_size = self.params['trainer']['accumulate_batch_size']
        eval_batch_size_multiplier = self.params['trainer']['eval_batch_size_multiplier']
        use_ema = self.params['trainer']['use_ema']

        train_loader = self.datasets['train'].get_loader( batch_size, True )
        test_loader = self.datasets['test'].get_loader( batch_size * eval_batch_size_multiplier , False )

        optimizer = self.get_optimizer( **self.params['trainer']['optimizer'] )

        if use_ema :
            ema = ModelEMA( self.model['net'] )
        else :
            ema = None

        if api :
            api.reset()
            api.add_cfg('nepoch', nepoch)
            api.add_cfg('keys', ['train_loss', 'test_acc'])
            api.send('started')

        self.ni = 0 
        self.nb = len(train_loader) 
        self.batch_size = batch_size 
        self.accumulate = accumulate_batch_size / batch_size
        self.last_opt_step = -1

        for epoch in range( nepoch ):

            print(f'Epoch {epoch+1}/{nepoch}')
            ave_loss = self.train_step( epoch, train_loader, optimizer, ema )
            acc = self.eval_step( test_loader, ema )

            if api :
                api.add_item({'train_loss': ave_loss, 'test_acc': acc, 'epoch': epoch})
                api.send('running' if epoch < nepoch-1 else 'done' )
