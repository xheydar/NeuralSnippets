import init

import os

import torch
import numpy as np
from torch import optim
from tqdm import tqdm

from datasets import datasets
#from noise_scheduler import NoiseScheduler
from diffusion_tools import DiffusionTools
from diffusion_transform import DiffusionTransform
from batch_generator import BatchGenerator

import model

from matplotlib import pyplot as pp
pp.ion()

class module :
    def __init__( self ):

        self.timesteps = 300
        self.batch_size = 64
        self.image_size = 64

        self.dataset = datasets['cifar10']('../data', image_size=self.image_size)

        use_cuda = torch.cuda.is_available()
        device_name = "cuda" if use_cuda else "cpu"
        self.device = torch.device( device_name )

    def setup_diffusion( self ):
        self.diffusion_tools = DiffusionTools(num_steps=self.timesteps, img_size=self.image_size, device=self.device)

    def build_batches( self ):
        transform = DiffusionTransform( self.diffusion_tools, device=self.device )
        self.batches = BatchGenerator( self.dataset.dataset, self.batch_size, transform=transform, randomize=True )

    def load_model( self ):
        self.model = {}
        self.model['unet'] = model.UNet().to(self.device)
        self.model['loss'] = model.Loss().to(self.device)

    def do_stuff( self ):
        x_noisy, noise, t, y = self.batches[10]

        out = self.model['unet']( x_noisy, t )


        return




        x = self.batches[10].to(self.device)

        t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device).long()

        x_noisy, noise = self.noise_scheduler.forward_diffusion_sample( x, t )
        noise_pred = self.model['unet'](x_noisy, t)
        loss = self.model['loss']( noise, noise_pred )

        print( loss )

        print( loss.item() )

        #t = torch.Tensor([50]).type(torch.int64)
        #image, noise = self.noise_scheduler.forward_diffusion_sample( batch, t )

        #image = self.dataset.reverse_transform( image[0] )
        #pp.imshow(image)

    @torch.no_grad()
    def sample_plot_image( self ):
        # Sample noise
        img_size = self.image_size
        img = torch.randn((1, 3, img_size, img_size), device=device)
        #plt.figure(figsize=(15,15))
        #plt.axis('off')
        num_images = 10
        stepsize = int(self.timesteps/num_images)

        for i in range(0,self.timesteps)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            noise_pred = self.model['unet']( img, t )
            img = self.noise_scheduler.sample_timestep(img, t, noise_pred)
            #if i % stepsize == 0:
            #    plt.subplot(1, num_images, i/stepsize+1)
            #    show_tensor_image(img.detach().cpu())
        #plt.show()

    def train( self, num_epoch=100 ):
        optimizer = optim.Adam(self.model['unet'].parameters(), lr=0.001)
        dataloader = torch.utils.data.DataLoader(dataset=self.dataset.dataset, batch_size=self.batch_size, shuffle=True)

        if not os.path.exists('./snapshots') : 
            os.makedirs('./snapshots')

        for epoch in range(num_epoch):
            print("Epoch : %d/%d" % ( epoch+1, num_epoch ) )
            loss_values = []
            for batch_idx, batch in tqdm(enumerate(dataloader)):

                optimizer.zero_grad()

                x = batch[0].to(self.device)
                t = torch.randint(0, self.timesteps, (len(x),), device=self.device).long()
                
                x_noisy, noise = self.noise_scheduler.forward_diffusion_sample( x, t, device=self.device )

                noise_pred = self.model['unet'](x_noisy, t)
                loss = self.model['loss']( noise, noise_pred )

                loss_values.append( loss.item() )

                loss.backward()
                optimizer.step()

            print( np.mean(loss_values) )
            state_dict = self.model['unet'].state_dict()
            torch.save({'state_dict':state_dict}, './snapshots/model_snapshot_%d.pt' % (epoch+1))

if __name__=="__main__" :

    m = module()
    m.load_model()
    m.train(num_epoch=300)





