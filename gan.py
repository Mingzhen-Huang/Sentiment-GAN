from fastai import *
from fastai.text import *
from tqdm import tqdm
from util import *
# import torch.nn as nn
class TextDicriminator(nn.Module):
    def __init__(self,encoder, nh):
        super().__init__()
        #encoder
        self.encoder = encoder
        #classifier
        layers = []
        layers+=bn_drop_lin(nh,1,p=0.15,actn=nn.Sigmoid())
        # layers+=SelfAttention(nh)
        layers += [nn.BatchNorm1d(1)]
        layers+=bn_drop_lin(nh,1,p=0.4,actn=nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
    
    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(0,2,1), (1,)).view(bs,-1)
    
    def forward(self, inp,y=None):
        _, outputs = self.encoder(inp)
        output = outputs[-1]
        bs,sl,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[:,-1], mxpool, avgpool], 1)
        out = self.layers(x)
        return out


