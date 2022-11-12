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
#import matplotlib.pyplot as plt

from models import Generator, Discriminator, Loss
from datasets import datasets


cudnn.benchmark = True

#set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#loading the dataset


mnist = datasets['mnist']('../data', train=True)
dataloader = torch.utils.data.DataLoader(mnist.dataset, batch_size=64,
                                         shuffle=True, num_workers=2)

#checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#number of channels in image(since the image is grayscale the number of channels are 1)
nc=1

# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

netG = Generator( nc=nc, nz=nz, ngf=ngf ).to(device)
netG.apply(weights_init)


netD = Discriminator( nc=nc, ndf=ndf ).to(device)
netD.apply(weights_init)
#netD.load_state_dict(torch.load('weights/netD_epoch_99.pth'))
#print(netD)

criterion = Loss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 25

fake = netG(fixed_noise)
vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (0), normalize=True)


# Commented out IPython magic to ensure Python compatibility.
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device, dtype=torch.float32)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                   % (epoch, niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    #vutils.save_image(real_cpu,'output/real_samples.png' ,normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch+1), normalize=True)        

    torch.save(netG.state_dict(), 'weights/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))

#num_gpu = 1 if torch.cuda.is_available() else 0

# load the models



#D = Discriminator(ngpu=1).eval()
#G = Generator(ngpu=1).eval()

# load weights
#D.load_state_dict(torch.load('weights/netD_epoch_99.pth'))
#G.load_state_dict(torch.load('weights/netG_epoch_99.pth'))
#if torch.cuda.is_available():
#    D = D.cuda()
#    G = G.cuda()

#batch_size = 25
#latent_size = 100

#fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
#if torch.cuda.is_available():
#    fixed_noise = fixed_noise.cuda()
#fake_images = G(fixed_noise)


# z = torch.randn(batch_size, latent_size).cuda()
# z = Variable(z)
# fake_images = G(z)

#fake_images_np = fake_images.cpu().detach().numpy()
#fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
#R, C = 5, 5
#for i in range(batch_size):
#    plt.subplot(R, C, i + 1)
#    plt.imshow(fake_images_np[i], cmap='gray')
#plt.show()

