import torch 
import torchvision
import numpy as np
import platform
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from easydict import EasyDict as edict

from dataset import dataset
import model


cfg = edict()
cfg.dataset = edict();

if platform.system() == "Darwin":
    cfg.dataset.root = '/Users/heydar/Work/void/cache'
else :
    cfg.dataset.root = '/home/heydar/cache'


class train :
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

        inputs = torch.cat( inputs, dim=0 )

        outputs = self.model['net']( inputs ) 

        print( outputs.shape )
        
    def train_step( self, train_loader, optimizer ):

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

            running_loss += float(loss)
            data_count += len(inputs)

        print( "Average loss :", running_loss / data_count )

    def eval_step( self, test_loader ):

        self.model['net'].eval()

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

        print("Eval acc : ", corrects / total)


    def train( self, nepoch=10 ):


        train_loader = self.datasets['train'].get_loader( 16, True )
        test_loader = self.datasets['test'].get_loader( 16, False )

        optimizer = optim.SGD( self.model['net'].parameters(), lr=0.001, momentum=0.9)

        for epoch in range( nepoch ):

            self.train_step( train_loader, optimizer )
            self.eval_step( test_loader )


if __name__=="__main__" :
    t = train()
    t.load_dataset()
    t.load_model()
    t.train()
