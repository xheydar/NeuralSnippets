import torch
import numpy as np

class BatchGenerator :
    def __init__( self, dataset, batch_size, transform, randomize=False ):
        self.dataset = dataset
        self.transform = transform
        ndata = len(self.dataset)

        inds = np.arange(ndata)

        if randomize :
            np.random.shuffle(inds)

        chunks = [ inds[i:i+batch_size] for i in range(0,ndata,batch_size) ]
        self.chunks = chunks

    def __len__( self ):
        return len(self.chunks)

    def __getitem__( self, idx ):
        chunk = self.chunks[idx]

        data = []
        labels = []

        for cidx in chunk :
            X,Y = self.dataset[cidx]

            X = X.unsqueeze(0)

            data.append( X )
            labels.append( Y )

        return self.transform( data, labels )




