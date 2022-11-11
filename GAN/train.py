import init

import sys
import numpy as np
from itertools import chain

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

    def build_model( self, weights = None ):
        self.latent_size = 10

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['encoder'] = model.Encoder( self.latent_size ).to(self.device)
        self.model['sampler'] = model.Sampler()
        self.model['generator'] = model.Generator( self.latent_size, self.dataset.shape ).to(self.device)
        self.model['discriminator'] = model.Discriminator( self.dataset.shape ).to(self.device)
        self.model['pretrain_loss'] = model.PretrainLoss().to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

    def dev_test_encoder( self ):
        X = torch.randn((128,1,28,28))

        out = self.model['encoder'](X)

        print( out.shape )


    def dev_test_generator( self ):
        X = torch.randn((128,self.latent_size))

        out = self.model['generator'](X)

        print( out.shape )

    def dev_test_discriminator( self ):
        X = torch.randn((128,1,28,28))

        out = self.model['discriminator'](X)

    def dev_autoencoder( self ):
        X = torch.randn((128,1,28,28))

        mu, logvar = self.model['encoder'](X)
        z = self.model['sampler']( mu, logvar )
        g = self.model['generator'](z)

        l = self.model['pretrain_loss']( X, mu, logvar, g )

        print(l)


    def pretrain(self, num_epoch):
        indim = self.dataset.dim

        optimizer = optim.Adam( chain( self.model['encoder'].parameters(),
                                       self.model['sampler'].parameters(),
                                       self.model['generator'].parameters()) )

        for epoch in range(num_epoch):
            train_loss = 0
            for batch_idx, (X, _) in tqdm(enumerate(self.data_loader)):
                X = X.to(self.device)
                optimizer.zero_grad()

                mu, logvar = self.model['encoder'](X)
                z = self.model['sampler']( mu, logvar )
                g = self.model['generator'](z)

                loss = self.model['pretrain_loss']( X, mu, logvar, g )
        
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
        
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.data_loader.dataset)))

        try :
            state_dict = self.model['network'].module.state_dict()
        except AttributeError:
            state_dict = self.model['network'].state_dict()

        torch.save({'state_dict':state_dict}, 'pretrain_model_%s.pt' % (self.dset))


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

    def pre_train( self, num_epoch=10 ):
        lr = 0.001
        discriminator_optim = optim.Adam( self.model['discriminator'].parameters(), lr=lr )
        generator_optim = optim.Adam( self.model['generator'].parameters(), lr=lr )


        for epoch in range(num_epoch):
            train_loss = 0

            D_losses = []
            G_losses = []

            print("Epoch %d/%d" % ( epoch+1, num_epoch ))

            for batch_idx, (X, _) in tqdm(enumerate(self.data_loader)):
                X = X.to(self.device)

                D_l = self.discriminator_train( X, discriminator_optim )

                D_losses.append( D_l )

            for batch_idx, (X, _) in tqdm(enumerate(self.data_loader)):
                X = X.to(self.device)

                G_l = self.generator_train( X, generator_optim )

                G_losses.append( G_l )
        
            print(' ==> Epoch : %d, D Average Loss : %g, G Average Loss : %g' % (epoch+1, np.mean(D_losses), np.mean(G_losses) ) )


    def train( self, num_epoch=10 ):
        indim = self.dataset.dim

        lr = 0.0002
        discriminator_optim = optim.SGD( self.model['discriminator'].parameters(), lr=lr )
        generator_optim = optim.Adam( self.model['generator'].parameters(), lr=lr )


        for epoch in range(num_epoch):
            train_loss = 0

            D_losses = []
            G_losses = []

            print("Epoch %d/%d" % ( epoch+1, num_epoch ))

            for batch_idx, (X, _) in tqdm(enumerate(self.data_loader)):
                X = X.to(self.device)

                D_l = self.discriminator_train( X, discriminator_optim )
                G_l = self.generator_train( X, generator_optim )

                D_losses.append( D_l )
                G_losses.append( G_l )

        
            print(' ==> Epoch : %d, D Average Loss : %g, G Average Loss : %g' % (epoch+1, np.mean(D_losses), np.mean(G_losses) ) )
#
#        try :
#            state_dict = self.model['network'].module.state_dict()
#        except AttributeError:
#            state_dict = self.model['network'].state_dict()

        state_dict = {}
        state_dict['discriminator'] = self.model['discriminator'].state_dict()
        state_dict['generator'] = self.model['generator'].state_dict()

        torch.save({'state_dict':state_dict}, 'model_%s.pt' % (self.dset))

if __name__=="__main__" :
    dset = sys.argv[1]
    stage = sys.argv[2]

    m = module( dset )

    if stage == "pretrain" :
        m.build_model()
        m.pretrain(50)
    elif stage == "train" :
        pass
    elif stage == "train_raw" :
        pass
