from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np

class validate :
    def __init__( self ):
        self._data_keys = ['test_acc']

    @property 
    def data_keys( self ):
        return deepcopy( self._data_keys )

    def __call__( self, test_loader, model, device ):
        
        model.eval()

        corrects = 0;
        total = 0

        for idx, data in tqdm(enumerate(test_loader), 0):
            inputs, labels = data 

            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model( inputs )
            pred = torch.argmax( pred, dim=1 )

            pred = pred.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            inds = np.where( pred == labels )[0]

            corrects += len(inds)
            total += len(labels)

        out = {}
        out['test_acc'] = corrects / total

        return out

