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
    def __init__(self, input_size, sentence_len, vocab_size, hidden_size, num_layers, embedding_dim, dropout):
        super(Generator, self).__init__()

        self.noise = torch.rand(input_size, hidden_size)
        self.vocab_size = vocab_size
        self.model = nn.Sequential(
            nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first = True,  num_layers = num_layers, dropout = dropout),
            nn.Linear(input_size, sentence_len),
            nn.Tanh(),
        )

    def forward(self):     
        result = self.model(self.noise).int()
        mask = torch.where((result > 0 and result < self.vocab_size), torch.tensor([1.]), torch.tensor([0.]))
        return result * mask


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, embeddings, dropout):
        super(Discriminator, self).__init__()        
        self.embeddings = embeddings

        self.model = nn.Sequential(
<<<<<<< HEAD
            nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first = True,  num_layers = num_layers, dropout = dropout)[:,-1,:],
            nn.Linear(hidden_size, 1),
=======
            nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout = dropout),
            nn.Linear(input_size, 1),
>>>>>>> ce7c32fae9ec96a998e43f8419a1c368448849e3
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.long)
        # mask = torch.where(inputs > 0, torch.tensor([1.]), torch.tensor([0.]))
        word_embed = self.embeddings(inputs)
        return self.model(word_embed)
