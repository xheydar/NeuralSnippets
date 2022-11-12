from __future__ import print_function

import init


import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from itertools import chain
#import matplotlib.pyplot as plt

from models import Encoder, Sampler, Generator, Discriminator, Loss, LossVAE
from datasets import datasets
from batch_generator import BatchGenerator

cudnn.benchmark = True

#set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class module :
    def __init__( self ):
        self.mnist = datasets['mnist']('../data', train=True)

    def build_batches( self ):
        self.batches = BatchGenerator( self.mnist.dataset, batch_size=64, randomize=True )
        
    def build_model( self ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #number of channels in image(since the image is grayscale the number of channels are 1)
        nc=1

        # input noise dimension
        nz = 100
        self.nz = nz
        # number of generator filters
        ngf = 64
        #number of discriminator filters
        ndf = 64

        self.model = {}
        
        self.model['encoder'] = Encoder( nc=nc, nz=nz, ngf=ngf ).to(self.device)
        self.model['sampler'] = Sampler().to(self.device)
        self.model['loss_vae'] = LossVAE().to(self.device)

        self.model['netG'] = Generator( nc=nc, nz=nz, ngf=ngf ).to(self.device)
        self.model['netG'].apply(weights_init)

        self.model['netD'] = Discriminator( nc=nc, ndf=ndf ).to(self.device)
        self.model['netD'].apply(weights_init)

        self.model['loss'] = Loss().to(self.device)

    def do_stuff( self ):
        X = self.batches[10]

        mu, logvar = self.model['encoder'](X)
        z = self.model['sampler'](mu,logvar)
        c = self.model['netG'](z)

        print( X[0] )
        print( c[0] )

        print( self.model['loss_vae']( c, mu, logvar, c ) )

    def pretrain( self, num_epoch=25 ): 
        optimizer = optim.Adam(chain( self.model['encoder'].parameters(),
                                      self.model['sampler'].parameters(),
                                      self.model['netG'].parameters()) )

        dataloader = torch.utils.data.DataLoader( self.mnist.dataset, batch_size=64,
                                                  shuffle=True, num_workers=2 )

        for epoch in range(num_epoch):
            for i, (X, _ ) in enumerate(dataloader):
                X = X.to(self.device)

                optimizer.zero_grad()

                mu, logvar = self.model['encoder'](X)
                z = self.model['sampler'](mu,logvar)
                c = self.model['netG'](z)

                loss = self.model['loss_vae']( c, mu, logvar, c )
                loss.backward()
                optimizer.step()

                L = output.mean().item()

                print(epoch+1, i+1, L)




    def train( self, num_epoch=25 ):
        optimizerD = optim.Adam(self.model['netD'].parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.model['netG'].parameters(), lr=0.0002, betas=(0.5, 0.999))

        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)
        real_label = 1
        fake_label = 0

        fake = self.model['netG'](fixed_noise)
        vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (0), normalize=True)

        dataloader = torch.utils.data.DataLoader( self.mnist.dataset, batch_size=64,
                                                  shuffle=True, num_workers=2 )


        # Commented out IPython magic to ensure Python compatibility.
        for epoch in range(num_epoch):
            for i, data in enumerate(dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
 
                self.model['netD'].zero_grad()
            
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=self.device, dtype=torch.float32)

                output = self.model['netD'](real_cpu)
                errD_real = self.model['loss'](output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device, dtype=torch.float32)
                fake = self.model['netG'](noise)
                label.fill_(fake_label)
                output = self.model['netD'](fake.detach())
                errD_fake = self.model['loss'](output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                
                self.model['netG'].zero_grad()
                label.fill_(real_label) 
                output = self.model['netD'](fake)
                errG = self.model['loss'](output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                   % (epoch, num_epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            #vutils.save_image(real_cpu,'output/real_samples.png' ,normalize=True)
            fake = self.model['netG'](fixed_noise)
            vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch+1), normalize=True)        

        #torch.save(netG.state_dict(), 'weights/netG_epoch_%d.pth' % (epoch))
        #torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))





if __name__=="__main__" :
    m = module()
    m.build_model()
    m.pretrain(5)
    m.train(25)
