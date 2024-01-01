from torch import nn 
import torch

class DeepSDFLoss:

    def __init__(self, delta, sd):
        self.mae = nn.L1Loss()
        self.delta = delta
        self.sd = sd

    def __call__(self, yhat, y, latent):
        l = self.mae(torch.clamp(yhat, -self.delta, self.delta), torch.clamp(y, -self.delta, self.delta))
        latent_norm = torch.pow(latent, 2).sum(dim=-1).mean()  * (1 / (self.sd ** 2))
        return l + latent_norm