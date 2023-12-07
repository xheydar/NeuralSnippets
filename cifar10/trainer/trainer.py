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

    def train_step( self, train_loader, optimizer, ema=None ):

        self.model['net'].train()

        running_loss = 0.0
        data_count = 0
        for idx, data in tqdm(enumerate( train_loader, 0 )):
            inputs, labels = data 

            inputs = inputs.to( self.device )
            labels = labels.to( self.device )

            optimizer.zero_grad()

            outputs = self.model['net']( inputs )

            loss = self.model['loss']( outputs, labels )

            loss.backward()
            optimizer.step()

            if ema :
                ema.update( self.model['net'] )


            running_loss += float(loss)
            data_count += len(inputs)

        return running_loss / data_count

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

    def train( self, nepoch=10, use_ema=True, api=None ):
        train_loader = self.datasets['train'].get_loader( 16, True )
        test_loader = self.datasets['test'].get_loader( 16, False )

        optimizer = self.get_optimizer('SGD', lr=0.001, momentum=0.9, decay=1e-5 )

        train_loss = []
        test_acc = []

        if use_ema :
            ema = ModelEMA( self.model['net'] )
        else :
            ema = None

        if api :
            api.reset()
            api.add_cfg('nepoch', nepoch)
            api.send('init')

        for epoch in range( nepoch ):

            print(f'Epoch {epoch+1}/{nepoch}')
            ave_loss = self.train_step( train_loader, optimizer, ema )
            acc = self.eval_step( test_loader, ema )

            train_loss.append( ave_loss )
            test_acc.append( acc )

            if api :
                api.add_item( {'train_loss': ave_loss, 'test_acc': acc, 'epoch': epoch })
                api.send('running')

        if api :
            api.send('done')
            

        out = {}
        out['train_loss'] = train_loss 
        out['test_acc'] = test_acc 

        return out

