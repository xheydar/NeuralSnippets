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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

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

    def build_model( self, pretrained=None ):
        self.latent_size = 50

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

        self.model = {}
        self.model['encoder'] = model.Encoder( 64, self.latent_size ).to(self.device)
        self.model['sampler'] = model.Sampler()
        self.model['generator'] = model.Generator( self.latent_size, 64 )
        self.model['discriminator'] = model.Discriminator( 64 ).to(self.device)
        self.model['pretrain_loss'] = model.PretrainLoss().to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

        
        self.model['generator'].apply( weights_init )
        self.model['discriminator'].apply( weights_init )

        if pretrained != None :
            print("Loading pretrained")
            model_data = torch.load( pretrained , map_location='cpu' )
            self.model['generator'].load_state_dict(  model_data['state_dict']['generator'], strict = True )

        self.model['generator'] = self.model['generator'].to(self.device)

    def dev_test_encoder( self ):
        X = torch.randn((128,1,28,28))

        out = self.model['encoder'](X)

        print( out.shape )


    def dev_test_generator( self ):
        X = torch.randn((128,self.latent_size,1,1))

        out = self.model['generator'](X)

        print( out.shape )

    def dev_test_discriminator( self ):
        X = torch.randn((128,1,28,28))

        out = self.model['discriminator'](X)

        print( out.shape )

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
        
            print('====> Epoch: {}/{} Average loss: {:.4f}'.format(epoch+1, num_epoch, train_loss / len(self.data_loader.dataset)))

        state_dict = {}
        
        #state_dict['discriminator'] = self.model['discriminator'].state_dict()
        state_dict['encoder'] = self.model['encoder'].state_dict()
        state_dict['generator'] = self.model['generator'].state_dict()

        torch.save({'state_dict':state_dict}, 'pretrain_model_%s.pt' % (self.dset))


    def discriminator_train( self, X, optim, z ):

        self.model['discriminator'].zero_grad();
        #optim.zero_grad()

        batch_size = len(X)
        x_real, y_real = X, torch.ones((batch_size,1)).to(self.device)

        D_real_out = self.model['discriminator']( x_real )
        D_real_loss = self.model['loss']( D_real_out, y_real )

        D_real_loss.backward()

        x_fake, y_fake = self.model['generator'](z), torch.zeros((batch_size,1)).to(self.device)

        D_fake_out = self.model['discriminator']( x_fake.detach() )
        D_fake_loss = self.model['loss']( D_fake_out, y_fake )

        D_fake_loss.backward()

        D_loss = D_real_loss + D_fake_loss

        optim.step()

        return float(D_loss.data.item())
        
    def generator_train( self, X, optim, z ):

        self.model['generator'].zero_grad()
        #optim.zero_grad()

        batch_size = len(X)

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
        beta1 = 0.5
        discriminator_optim = optim.Adam( self.model['discriminator'].parameters(), lr=lr, betas=(beta1, 0.999) )
        generator_optim = optim.Adam( self.model['generator'].parameters(), lr=lr, betas=(beta1, 0.999) )


        for epoch in range(num_epoch):
            train_loss = 0

            D_losses = []
            G_losses = []

            print("Epoch %d/%d" % ( epoch+1, num_epoch ))

            for batch_idx, (X, _) in tqdm(enumerate(self.data_loader)):
                X = X.to(self.device)

                batch_size = len(X)
                z = torch.randn((batch_size,self.latent_size,1,1)).to(self.device) 



                D_l = self.discriminator_train( X, discriminator_optim, z )
                G_l = self.generator_train( X, generator_optim, z )

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
        m.build_model(pretrained="pretrain_model_%s.pt" % ( dset ) )
        m.train(100)
    elif stage == "train_raw" :
        m.build_model()
        m.train(100)

