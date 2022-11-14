import torch
import torch.nn.functional as F

class NoiseScheduler :
    def __init__( self, timesteps, start, end ):
        self.timesteps = int(timesteps)
        self.start = start
        self.end = end
        self.betas = torch.linspace( self.start,
                                     self.end,
                                     self.timesteps )

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod( self.alphas, axis=0 )
        self.alphas_cumprod_prev = F.pad( self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt( 1.0 / self.alphas )
        self.sqrt_alphas_cumprod = torch.sqrt( self.alphas_cumprod )
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * ( 1.0 - self.alphas_cumprod_prev ) / ( 1.0 - self.alphas_cumprod )

    def get_index_from_list( self, vals, t, x_shape ):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample( self, x_0, t, device="cpu" ):

        noise = torch.randn_like( x_0 )
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) \
                * noise.to(device), noise.to(device)
