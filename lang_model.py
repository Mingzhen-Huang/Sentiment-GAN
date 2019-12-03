from fastai import *
from fastai.text import *
from fastai.callbacks.tracker import SaveModelCallback, EarlyStoppingCallback

models = {'AWD':AWD_LSTM, 'XL':TransformerXL}

#train language model with either AWD_LSTM or TransformerXL archs and generate preds
def train_lm(path,filename,model='AWD_LSTM',
             epochs=8,pretrained_fnames=None,preds=True):
    
    #get data after running preprocess
    print(f'loading data from {path}/{filename};')
    data_lm = load_data(path,filename, bs=64,bptt=70)
    
    if pretrained_fnames: pretrained_fnames = pretrained_fnames.split(',')
    learn = language_model_learner(data_lm,models[model],
                                   config=config,pretrained=False,
                                   pretrained_fnames=pretrained_fnames)
    print(f'training lm model {model}; pretrained from {pretrained_fnames};')
    
    #early stopping and saving at every epoch
    cb = [SaveModelCallback(learn),EarlyStoppingCallback(learn)]
    
    if pretrained_fnames:
        #layered training
        print(f'training lm model head;')
        learn.fit_one_cycle(1, 3e-3, moms=(0.8,0.7))
        print(f'saving lm model head to {path}/{filename}_head;')
        learn.save(filename+'_head')
        learn.unfreeze()
        
    print(f'training for {epochs} epochs')
    learn.fit_one_cycle(epochs, 3e-4, moms=(0.8,0.7),callbacks=cb)
    print(f'saving model to {path}/{filename}_finetuned')
    learn.save(filename+'_lm')

if __name__ == '__main__': 
    train_lm('/content/drive/My Drive','poems_tmp','AWD',8)