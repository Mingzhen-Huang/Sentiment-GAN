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
import pdb

class Generator(nn.Module):
    def __init__(self, input_size, sentence_len, vocab_size, hidden_size, num_layers, embedding_dim, dropout):
        super(Generator, self).__init__()

        self.noise = torch.rand(input_size, sentence_len, embedding_dim)
        self.vocab_size = vocab_size
        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_size, batch_first = True,  num_layers = num_layers, dropout = dropout)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self):    
        # pdb.set_trace()
        rnn_output,_ = self.gru(self.noise,None)
        rnn_output = torch.squeeze(rnn_output)
        dense_output = self.dense(rnn_output)*self.vocab_size
        result = torch.squeeze(torch.abs(dense_output).int())
        mask = torch.where(result < self.vocab_size, torch.tensor([1.]), torch.tensor([0.]))
        # pdb.set_trace()
        return result * mask


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, embeddings, dropout):
        super(Discriminator, self).__init__()        
        self.embeddings = embeddings

        self.gru = nn.GRU(input_size = embedding_dim, hidden_size = hidden_size, batch_first = True,  num_layers = num_layers, dropout = dropout)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, inputs):

        inputs = torch.tensor(inputs,dtype=torch.long)
        batch_size = inputs.shape[0]
        # pdb.set_trace()
        # mask = torch.where(inputs > 0, torch.tensor([1.]), torch.tensor([0.]))
        word_embed = self.embeddings(inputs)
        
        rnn_output,_ = self.gru(word_embed,None)
        rnn_output = torch.squeeze(rnn_output[:,-1,:])
        dense_output = self.dense(rnn_output)
        
        logits = nn.Sigmoid()(torch.squeeze(dense_output))
        return logits.view(batch_size, 1)
