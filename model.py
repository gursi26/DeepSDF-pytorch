from torch import nn 
import torch

class DeepSDF(nn.Module):

    def __init__(self, input_dim, layer_size = 512, dropout_p = 0.2):
        super(DeepSDF, self).__init__()
        self.dropout_p = dropout_p
        self.input_layer = self.create_layer_block(input_dim, layer_size)
        self.layer2 = self.create_layer_block(layer_size, layer_size)
        self.layer3 = self.create_layer_block(layer_size, layer_size)
        self.layer4 = self.create_layer_block(layer_size, layer_size - input_dim)
        self.layer5 = self.create_layer_block(layer_size, layer_size)
        self.layer6 = self.create_layer_block(layer_size, layer_size)
        self.layer7 = self.create_layer_block(layer_size, layer_size)
        self.layer8 = self.create_layer_block(layer_size, 1)

    def create_layer_block(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )

    def forward(self, latent_vec, coords):
        """
        latent_vec has shape [batch_size, z_dim]
        coords has shape [batch_size, num_coords, 3]
        """
        # latent_vec now has shape [batch_size, num_coords, z_dim], repeated on the middle axis
        latent_vec = latent_vec.unsqueeze(1).repeat(1, coords.shape[1], 1)

        x = torch.cat([latent_vec, coords], dim = -1)
        skip_x = x

        x = self.input_layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(torch.cat([x, skip_x], dim = -1)) # skip connection
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        # return has shape [batch_size, num_coords], where each element is the SDF
        # at the given input coordinate
        return x.squeeze(-1)