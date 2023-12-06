import torch 
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

class trainer :
    def __init__( self ):
        pass

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
                ema.update( self.model['train'] )


            running_loss += float(loss)
            data_count += len(inputs)

        return running_loss / data_count

    def eval_step( self, test_loader, ema ):

        if emal :
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

    def train( self, nepoch=10, use_ema=True ):
        train_loader = self.datasets['train'].get_loader( 16, True )
        test_loader = self.datasets['test'].get_loader( 16, False )

        optimizer = optim.SGD( self.model['net'].parameters(), lr=0.001, momentum=0.9)

        train_loss = []
        test_acc = []

        if use_ema :
            ema = ModelEMA( self.model['net'] )
        else :
            ema = None

        for epoch in range( nepoch ):

            print(f'Epoch {epoch+1}/{nepoch}')
            ave_loss = self.train_step( train_loader, optimizer )
            acc = self.eval_step( test_loader )

            train_loss.append( ave_loss )
            test_acc.append( acc )

        out = {}
        out['train_loss'] = train_loss 
        out['test_acc'] = test_acc 

        return out

