from fastai import *
from fastai.text import *
from fastai.callbacks.tracker import SaveModelCallback, EarlyStoppingCallback
from gan import *
from util import *
from sentiment_loss import *

import argparse
import logging

models = {'AWD':AWD_LSTM}

def train_lm(path,filename,model='AWD_LSTM',
             epochs=8,pretrained_fnames=None):
    
    #get data after running preprocess
    print(f'loading data from {path}/{filename};')
    data_lm = load_data(path,filename, bs=64,bptt=70)
    
    if pretrained_fnames: pretrained_fnames = pretrained_fnames.split(',')
    learn = language_model_learner(data_lm,models[model],
                                   config=None,pretrained=False,
                                   pretrained_fnames=pretrained_fnames)
    print(f'training lm model {model}; pretrained from {pretrained_fnames};')
    
    #early stopping and saving at every epoch
    cb = [SaveModelCallback(learn),EarlyStoppingCallback(learn)]
    
    if pretrained_fnames:
        #layered training
        print(f'training lm model head;')
        learn.fit_one_cycle(1, 3e-3, moms=(0.8, 0.7))
        print(f'saving lm model head to {path}/{filename}_head;')
        learn.save(filename+'_head')
        learn.unfreeze()
        
    print(f'training for {epochs} epochs')
    learn.fit_one_cycle(epochs, 3e-4, moms=(0.8,0.7),callbacks=cb)
    print(f'saving model to {path}/{filename}_finetuned')
    learn.save(filename+'_lm')


def train(gen, disc, epochs, trn_dl, val_dl, optimizerD, optimizerG, crit=None, first=True, senti_disc=None):
    gen_iterations = 0
    
    for epoch in range(epochs):
        gen.train(); disc.train()
        n = len(trn_dl)
        #train loop
        with tqdm(total=n) as pbar:
            for i, ds in enumerate(trn_dl):
                x, y = ds
                bs,sl = x.size()
                disc.eval(), gen.train()
                fake,_,_ = gen(x)
                gen.zero_grad()
                fake_sample =seq_gumbel_softmax(fake)
                with torch.no_grad():
                    gen_loss = reward = disc(fake_sample)

                    if senti_disc:
                        sentiment_loss = senti_disc.get(fake_sample)
                        print(sentiment_loss)

                    if crit: gen_loss = crit(fake,fake_sample,reward.squeeze(1))
                    gen_loss = gen_loss.mean()
                gen_loss.requires_grad_(True)
                gen_loss.backward()
                optimizerG.step()
                gen_iterations += 1
                d_iters = 3
                for j in range(d_iters):
                    gen.eval()
                    disc.train()
                    with torch.no_grad():
                        fake,_,_ = gen(x)
                        fake_sample = seq_gumbel_softmax(fake)
                    disc.zero_grad()
                    fake_loss = disc(fake_sample)
                    #fake_loss.requires_grad=True
                    real_loss = disc(y.view(bs,sl))
                    #real_loss.requires_grad=True
                    disc_loss = (fake_loss-real_loss).mean(0)
                    disc_loss.backward()
                    optimizerD.step()
                pbar.update()
        print(f'Epoch {epoch}:')
        print('Train Loss:')
        print(f'Loss_D {disc_loss.data.item()}; Loss_G {gen_loss.data.item()} Ppx {torch.exp(lm_loss(fake,y))}')
        print(f'D_real {real_loss.mean(0).view(1).data.item()}; Loss_D_fake {fake_loss.mean(0).view(1).data.item()}')
        disc.eval(), gen.eval()
        with tqdm(total=len(val_dl)) as pbar:
            for i, ds in enumerate(val_dl):
                with torch.no_grad():
                    x, y = ds
                    bs, sl = x.size()
                    fake, _, _ = gen(x)
                    fake_sample =seq_gumbel_softmax(fake)
                    gen_loss = reward = disc(fake_sample)
                    if crit: gen_loss = crit(fake, fake_sample, reward.squeeze(1))
                    gen_loss = gen_loss.mean()
                    fake_sample = seq_gumbel_softmax(fake)
                    fake_loss = disc(fake_sample)
                    real_loss = disc(y.view(bs, sl))
                    disc_loss = (fake_loss-real_loss).mean(0)
                pbar.update()
        print('Valid Loss:')
        print(f'Loss_D {disc_loss.data.item()}; Loss_G {gen_loss.data.item()} Ppx {torch.exp(lm_loss(fake,y))}')
        print(f'D_real {real_loss.mean(0).view(1).data.item()}; Loss_D_fake {fake_loss.mean(0).view(1).data.item()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--path', type=str, help="path of data", default='./data')
    parser.add_argument('--train_lm', action="store_true", default=False)
    parser.add_argument('--train_gan', action="store_true", default=False)
    parser.add_argument('--lm_epoch', type=int, default=8)
    parser.add_argument('--gan_epoch', type=int, default=8)
    parser.add_argument('--pretrain_lm', type=str,  default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if args.train_lm:
        # train a language model with awd-lstm
        train_lm(path,'poems_tmp','AWD',args.lm_epoch, args.pretrain_lm)

    if args.train_gan:
        data_lm = load_data(path, 'poems_tmp')
        trn_dl = data_lm.train_dl
        val_dl = data_lm.valid_dl
        learn = language_model_learner(data_lm, arch=AWD_LSTM)
        learn.load('poems_tmp_lm')

        encoder = deepcopy(learn.model[0])
        x, y = next(iter(trn_dl))
        outs = encoder(x)
        generator = deepcopy(learn.model) 
        generator.load_state_dict(learn.model.state_dict())

        disc = TextDicriminator(encoder, 400).cuda()
        out = disc(x)
        probs, raw_outputs, outputs = generator(x)
        optimizerD = optim.Adam(disc.parameters(), lr = 3e-4)
        optimizerG = optim.Adam(generator.parameters(), lr = 3e-3, betas=(0.7, 0.8))

        senti_disc = sentiment_loss(data_lm)

        disc.train()
        generator.train()

        train(generator, disc, args.gan_epoch, trn_dl, val_dl, optimizerD, optimizerG, first=False, senti_disc=senti_disc)
        learn.model.load_state_dict(generator.state_dict())
        learn.predict("Red",n_words=50)
        learn.save('poems_gan_gumbel')
