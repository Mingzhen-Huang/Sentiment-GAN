from fastai.text import *
from fastai.callbacks.tracker import SaveModelCallback, EarlyStoppingCallback

def lm_loss(input, target, kld_weight=0):
    sl, bs = target.size()
    sl_in,bs_in,nc = input.size()
    return F.cross_entropy(input.view(-1,nc), target.view(-1))

# def bn_drop_lin(n_in, n_out, bn=True, initrange=0.01,p=0, bias=True, actn=nn.LeakyReLU(inplace=True)):
#     layers = [nn.BatchNorm1d(n_in)] if bn else []
#     if p != 0: layers.append(nn.Dropout(p))
#     linear = nn.Linear(n_in, n_out, bias=bias)
#     if initrange:linear.weight.data.uniform_(-initrange, initrange)
#     if bias: linear.bias.data.zero_()
#     layers.append(linear)
#     if actn is not None: layers.append(actn)
#     return layers

def stats(tensor):return torch.mean(tensor),torch.std(tensor)

def seq_gumbel_softmax(input):
    samples = []
    bs,sl,nc = input.size()
    for i in range(sl): 
        samples.append(torch.multinomial(F.gumbel_softmax(input[:,i,:]),1))
    samples = torch.stack(samples).transpose(1,0).squeeze(2) 
    return samples