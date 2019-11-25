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
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, dropout):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
        )

    def forward(self, noise):       
        return self.model(noise).int()


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, embeddings, dropout):
        super(Discriminator, self).__init__()        
        self.embeddings = embeddings
        self.model = nn.Sequential(
            nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout),
            nn.Linear(input_size, 1),
            F.sigmoid(),
        )

    def forward(self, inputs):
        mask = torch.where(inputs > 0, torch.tensor([1.]), torch.tensor([0.]))
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        return self.model(word_embed)
