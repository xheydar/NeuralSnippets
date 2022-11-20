import torch

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

    def sample_timesteps( self, n ):
        return torch.randint( low=1, high=self.num_steps, size=(n,) )

    def noise_images( self, x, t ):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt( 1-self.alpha_hat[t])[:,None,None,None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample( self, model, n, labels, cfg_scale=3 ):


        #model.eval()

        with torch.no_grad() :
            x = torch.randn((n, self.n_channels,self.img_size, self.img_size)).to(self.device)

            for i in reversed(range(1,self.num_steps)) :
                t = (torch.ones(n) * i).long().to(self.device)
                print( i )
