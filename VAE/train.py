import init

import torch
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm

import model

class module :
    def __init__( self ):
        self.dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
        self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=128, shuffle=True)

    def build_model( self ):
        latent_size = 2
        layers = [784, 512, 256]

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['network'] = model.Net( layers, latent_size ).to(self.device)
        self.model['loss'] = model.Loss().to(self.device)
        
    def train( self, num_epoch=10 ):
        optimizer = optim.Adam(self.model['network'].parameters())

        for epoch in range(num_epoch):
            train_loss = 0
            for batch_idx, (data, _) in tqdm(enumerate(self.data_loader)):
                data = data.view(-1, 784).to(self.device)
                optimizer.zero_grad()
        
                mu, log_var, recon = self.model['network'](data)
                loss = self.model['loss'](data, mu, log_var, recon)
        
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
        
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.data_loader.dataset)))

        try :
            state_dict = self.model['network'].module.state_dict()
        except AttributeError:
            state_dict = self.model['network'].state_dict()

        torch.save({'state_dict':state_dict}, 'model.pt')


if __name__=="__main__" :
    m = module()
    m.build_model()
    m.train(100)
