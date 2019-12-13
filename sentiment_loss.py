from fastai import *
from fastai.text import *
from tqdm import tqdm
from util import *
import pandas as pd

class sentiment_loss():
    def __init__(self, data_lm, negative=True):
        super().__init__()

        df = pd.read_json('./data/imdb_sentiment_train_15k.jsonl', 'records', lines=True)
        df_test = pd.read_json('./data/imdb_sentiment_test.jsonl', 'records', lines=True)
        data_lm_sen = TextLMDataBunch.from_df('./data/', df, df_test, text_cols=0, label_cols=1)
        data_clas = TextClasDataBunch.from_df('./data/', df, df_test, vocab=data_lm_sen.train_ds.vocab, bs=64,
                                              text_cols=0, label_cols=1)
        model = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
        model.load_encoder('ft_enc')

        self.model = model.load('sentiment_disc')
        self.data_lm = data_lm
        self.negative = negative

    def get(self, fake_sample):
        batch_loss = []
        for s in fake_sample:
            text = self.data_lm.vocab.textify(s)
            loss = self.model.predict(text)[2][1]
            batch_loss.append(loss)
        if self.negative:
            return torch.mean(torch.tensor(batch_loss))
        else:
            return 1-torch.mean(torch.tensor(batch_loss))

    def pred(self, poem):
        return self.model.predict(poem)[2][1]
