import pickle
from tqdm import tqdm


print_interval = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--d-lr', type=float, help="learning rate of generator model", default=1)
    parser.add_argument('--g-lr', type=float, help="learning rate of discriminator model", default=1)
    parser.add_argument('--batch-size', type=int, help="size of batch", default=10)
    parser.add_argument('--epochs', type=int, help="num epochs", default=10)
    parser.add_argument('--embed-file', type=str, help="embedding location", default='./data/glove.6B.100D.txt')
    parser.add_argument('--embed-dim', type=int, help="size of embeddings", default=100)
    parser.add_argument('--hidden-size', type=int, help="size of hidden dimension", default=128)
    parser.add_argument('--d-steps', type=int, help="numbers of training discriminator for an epoch", default=10)
    parser.add_argument('--g-steps', type=int, help="numbers of training generator for an epoch", default=10)
    args = parser.parse_args()
 
    G = Generator()
    D = Discriminator()
    criterion = nn.BCELoss()  
    d_optimizer = optim.Adam(D.parameters(), lr=args.d_lr, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=optim_betas)
    for epoch in range(args.epochs):
        for d_index in range(args.d_steps):
            D.zero_grad()
     
            #  Train D on real
            d_real_data = None
            d_real_decision = None
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params
     
            #  Train D on fake
            d_gen_input = None
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D()
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     
     
        for g_index in range(args.g_steps):
            #  Train G on D's response 
            G.zero_grad()
     
            gen_input = None
            g_fake_data = G(gen_input)
            dg_fake_decision = D()
            g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # pretend all true
     
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
     
        if epoch % print_interval == 0:
            print()