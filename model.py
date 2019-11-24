import argparse
import os
import numpy as np
import math

import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, noise, labels):       
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        

        self.model = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, inputs, labels):
       
        return None
