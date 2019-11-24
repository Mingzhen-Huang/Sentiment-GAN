import pickle
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import Generator, Discriminator
import torch.optim as optim
import random
from torch.autograd import Variable



print_interval = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--d-lr', type=float, help="learning rate of generator model", default=0.001)
    parser.add_argument('--g-lr', type=float, help="learning rate of discriminator model", default=0.001)
    parser.add_argument('--n-layer', type=int, help='number of GRU layers', default=1)
    parser.add_argument('--batch-size', type=int, help="size of batch", default=10)
    parser.add_argument('--epochs', type=int, help="num epochs", default=10)
    parser.add_argument('--embed-file', type=str, help="embedding location", default='./data/glove.6B.100D.txt')
    parser.add_argument('--embed-dim', type=int, help="size of embeddings", default=100)
    parser.add_argument('--drop-out',type=float,help='drop out of GRU', default=0)
    parser.add_argument('--hidden-size', type=int, help="size of hidden dimension", default=128)
    parser.add_argument('--d-steps', type=int, help="numbers of training discriminator for an epoch", default=10)
    parser.add_argument('--g-steps', type=int, help="numbers of training generator for an epoch", default=10)
    parser.add_argument('--adam-beta', type=tuple, help='beta1 for Adam optimizers', default=(0.5,0.999))
    parser.add_argument('--weight-decay', type=float, help='weight decay for Adam optimizers', default=0)
    parser.add_argument('--save-path', type=str, help='path to save models', default='models/')
    args = parser.parse_args()

    manualSeed = 999
    model_num = 0
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # todo input parameters
    G = Generator(args.batch_size, args.hidden_size, args.n_layer, args.embed_dim, args.drop_out)
    D = Discriminator(args.batch_size, args.hidden_size, args.n_layer, args.embed_dim, args.drop_out)

    criterion = nn.BCELoss()  
    d_optimizer = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.adam_beta, weight_decay=args.weight_decay)
    g_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.adam_beta, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        total_d_real_loss = 0
        total_d_fake_loss = 0
        total_g_loss = 0

        D.train()

        for d_index in range(args.d_steps):

            for data in range(args.batch_size):
                #  Train D on real
                D.zero_grad()
                d_real_data = None
                d_real_label = D(d_real_data)
                d_real_error = criterion(d_real_label, Variable(torch.ones(1)))  # ones = true
                total_d_real_loss += d_real_error
                d_real_error.backward()  # compute/store gradients, but don't change params
                #  Train D on fake
                d_gen_input = None
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_label = D(d_fake_data)
                d_fake_error = criterion(d_fake_label, Variable(torch.zeros(1)))  # zeros = fake
                total_d_fake_loss += d_fake_error
                d_fake_error.backward()
                d_optimizer.step()

        avg_d_real_loss = total_d_real_loss/(args.batch_size*args.d_steps)
        avg_d_fake_loss = total_d_fake_loss / (args.batch_size * args.d_steps)

        G.train()
        for g_index in range(args.g_steps):
            #  Train G on D's response
            length = None  # todo sentence length to generate
            for data in range(args.batch_size):
                G.zero_grad()

                gen_input = torch.randn(args.batch_size, length)


                g_fake_data = G(gen_input)
                dg_fake_label = D(g_fake_data)
                g_error = criterion(dg_fake_label, Variable(torch.ones(1)))  # pretend all true
                total_g_loss+=g_error
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

        avg_g_loss = total_g_loss/args.batch_size * args.g_steps

        if epoch % print_interval == 0:
            torch.save(D.state_dict(), args.save_path + 'Discriminator_model_' + str(model_num))
            torch.save(G.state_dict(), args.save_path + 'Generator_model_' + str(model_num))
            model_num += 1
            print('avg_Discriminator_real_loss:',avg_d_real_loss)
            print('avg_Discriminator_fake_loss:', avg_d_fake_loss)
            print('avg_Generator_loss:', avg_g_loss)
