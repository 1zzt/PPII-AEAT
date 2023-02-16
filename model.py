import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg


class Highway(nn.Module):

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
        return x  

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

class Encoder(nn.Module):
    def __init__(self, n_hidden=128, n_latent=128, n_inputs=200):
        super(Encoder, self).__init__()
        self.n_hidden = n_hidden
        self.n_latent = n_latent


        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 1200),  
            RMSNorm(1200), 
            nn.GELU(),

            nn.Linear(1200, 800),  
            RMSNorm(800), 
            nn.GELU(),

            nn.Linear(800, 400),  
            RMSNorm(400), 
            nn.GELU(),

            nn.Linear(400, 256),  
            RMSNorm(256), 
            nn.GELU(),



            nn.Linear(256, 128),  
        )


    def forward(self, x):
        x = self.encoder(x)
        return x 


class Decoder(nn.Module):
    def __init__(self, n_hidden=128, n_latent=128, n_inputs=200):
        super(Decoder, self).__init__()
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        
        self.decoder =nn.Sequential(
            nn.Linear(128,256),   
            RMSNorm(256),
            nn.GELU(), 
            
            nn.Linear(256,400),   
            RMSNorm(400),
            nn.GELU(), 

            nn.Linear(400,800),   
            RMSNorm(800),
            nn.GELU(), 

            nn.Linear(800,1200),   
            RMSNorm(1200),
            nn.GELU(), 
            
            
            Highway(2,  1200),
            nn.Linear(1200, n_inputs)
        )

    def forward(self, x):
        x = self.decoder(x)

        return x 


class AES_FP_DP(nn.Module):
    def __init__(self, n_hidden=128, n_latent=128, n_inputs_fp=200, n_inputs_dp=200 ):
        super(AES_FP_DP, self).__init__()
            
            
        self.n_hidden = n_hidden
        self.n_latent = n_latent

        self.encoder_fp = Encoder(n_hidden, n_latent, n_inputs_fp)
        self.decoder_fp1 = Decoder(n_hidden, n_latent, n_inputs_fp)
        self.decoder_fp2 = Decoder(n_hidden, n_latent, n_inputs_fp)

        self.encoder_dp = Encoder(n_hidden, n_latent, n_inputs_dp)
        self.decoder_dp1 = Decoder(n_hidden, n_latent, n_inputs_dp)
        self.decoder_dp2 = Decoder(n_hidden, n_latent, n_inputs_dp)

        
        self.gate = nn.Sequential(
            nn.Linear(self.n_latent*2, self.n_latent), nn.Sigmoid()
            )
        self.predict  =  nn.Sequential(
            RMSNorm(self.n_latent*2), 
            nn.Linear(self.n_latent*2, 1),  

        )

    def forward_once(self, g, encoder, decoder1, decoder2):
        z = encoder(g)
        ## Decoders (Phase 1)
        ae1 = decoder1(z)
        ae2 = decoder2(z)
        ## Encode-Decode (Phase 2)
        z2 = encoder(ae1)
        ae2ae1 = decoder2(z2)
        return z, ae1, ae2, ae2ae1


    def forward(self, fp, dp):

        z_fp, ae1_fp, ae2_fp, ae2ae1_fp = self.forward_once(fp, self.encoder_fp, self.decoder_fp1, self.decoder_fp2)
        z_dp, ae1_dp, ae2_dp, ae2ae1_dp = self.forward_once(dp, self.encoder_dp, self.decoder_dp1, self.decoder_dp2)

        encoder_output = torch.cat( (z_fp, z_dp), 1 )
        pred = self.predict(encoder_output)

        return ae1_fp, ae2_fp, ae2ae1_fp, ae1_dp, ae2_dp, ae2ae1_dp, pred

def ae_loss_function(Wt, Wt1p, Wt2p, Wt2dp, n):
    """
        :param Wt: ground truth sequence
        :param Wt1p: AE1 decoder output
        :param Wt2p: AE2 decoder output
        :param Wt2dp: AE1 encoder output => AE2 decoder
        :param n: Training epochs
    """
    loss_AE1 = (1 / n) * F.mse_loss(Wt, Wt1p) + (1 - (1 / n)) * F.mse_loss(Wt, Wt2dp)
    loss_AE2 = (1 / n) * F.mse_loss(Wt, Wt2p) - (1 - (1 / n)) * F.mse_loss(Wt, Wt2dp)
    return loss_AE1 + loss_AE2