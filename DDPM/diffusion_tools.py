import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

class DiffusionTools :
    def _noise_schedule( self ):
        return torch.linspace( self.beta_start, self.beta_end, self.num_steps )

    def __init__( self, num_steps=1000, beta_start=1e-4, beta_end=0.02,
                        img_size=256, n_channels=3, device=torch.device('cpu') ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.img_size = img_size
        self.n_channels = n_channels
        self.device = device

        self.beta = self._noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod( self.alpha, dim=0 )

        self.reverse_transform = transforms.Compose([
            transforms.Lambda( lambda t : (t.clamp(-1,1)+1)/2 ),
            transforms.Lambda( lambda t : t.permute(1,2,0) ),
            transforms.Lambda( lambda t : t * 255 ),
            transforms.Lambda( lambda t : t.numpy().astype(np.uint8))
        ])

    def sample_timesteps( self, n ):
        return torch.randint( low=1, high=self.num_steps, size=(n,) )

    def noise_images( self, x, t ):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt( 1-self.alpha_hat[t])[:,None,None,None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample( self, model, n, labels, cfg_scale=3 ):


        #model.eval()

        out = []

        with torch.no_grad() :
            x = torch.randn((n, self.n_channels,self.img_size, self.img_size)).to(self.device)

            patch = self.reverse_transform( x[0] )
            cv2.imwrite("images/patch_%04d.jpg" % (self.num_steps), patch )

            for i in reversed(range(1,self.num_steps)) :
                print( i )
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                patch = self.reverse_transform( x[0] )
                cv2.imwrite("images/patch_%04d.jpg" % (i), patch )

        return out
