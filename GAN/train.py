import init

import sys

import torch
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from datasets import datasets
import model

class module :
    def __init__( self, dset ):
        self.dset = dset

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])


        self.dataset = datasets[self.dset](root='../data', train=True, transform=transform)
        self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset.dataset, batch_size=128, shuffle=True)

    def build_model( self ):
        indim = self.dataset.dim
        self.latent_size = 10

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['generator'] = model.Generator( self.latent_size, indim ).to(self.device)
        self.model['discriminator'] = model.Discriminator( indim ).to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

    def discriminator_train( self, X, optim ):
        optim.zero_grad()

        batch_size = len(X)
        x_real, y_real = X, torch.ones((batch_size,1)).to(self.device)

        D_real_out = self.model['discriminator']( x_real )
        D_real_loss = self.model['loss']( D_real_out, y_real )

        z = torch.randn((batch_size,self.latent_size)).to(self.device) 
        x_fake, y_fake = self.model['generator'](z), torch.zeros((batch_size,1)).to(self.device)

        D_fake_out = self.model['discriminator']( x_fake )
        D_fake_loss = self.model['loss']( D_fake_out, y_fake )

        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        optim.step()

        return float(D_loss.data.item())
        
    def generator_train( self, X, optim ):
        optim.zero_grad()

        batch_size = len(X)

        z = torch.randn((batch_size,self.latent_size)).to(self.device)
        y = torch.ones((batch_size,1)).to(self.device)

        G_output = self.model['generator'](z)
        D_output = self.model['discriminator'](G_output)
        G_loss = self.model['loss']( D_output, y )

        G_loss.backward()
        optim.step()

        return float(G_loss.data.item())

    def train( self, num_epoch=10 ):
        indim = self.dataset.dim

        lr = 0.0002
        discriminator_optim = optim.Adam( self.model['discriminator'].parameters(), lr=lr )
        generator_optim = optim.Adam( self.model['generator'].parameters(), lr=lr )


        for epoch in range(num_epoch):
            train_loss = 0

            D_losses = []
            G_losses = []

            for batch_idx, (X, _) in tqdm(enumerate(self.data_loader)):
                X = X.view(-1, indim).to(self.device)

                D_l = self.discriminator_train( X, discriminator_optim )
                G_l = self.generator_train( X, generator_optim )

                D_losses.append( D_l )
                G_losses.append( G_l )

        
            print(' ==> Epoch : %d, D Average Loss : %g, G Average Loss : %g' % (epoch, np.mean(D_losses), np.mean(G_losses) ) )
#
#        try :
#            state_dict = self.model['network'].module.state_dict()
#        except AttributeError:
#            state_dict = self.model['network'].state_dict()
#
#        torch.save({'state_dict':state_dict}, 'model_%s.pt' % (self.dset))

if __name__=="__main__" :
    m = module( sys.argv[1] )
    m.build_model()
    m.train(100)
