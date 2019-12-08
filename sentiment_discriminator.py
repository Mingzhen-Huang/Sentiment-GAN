from fastai import *
from fastai.text import *
from tqdm import tqdm
from util import *
import pandas as pd

df = pd.read_json('./data/imdb_sentiment_train_15k.jsonl', 'records', lines=True)
df_test = pd.read_json('./data/imdb_sentiment_test.jsonl', 'records', lines=True)

# print(df)

data_lm = TextLMDataBunch.from_df('./data/', df, df_test, text_cols=0, label_cols=1)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)
learn.save_encoder('ft_enc')


data_clas = TextClasDataBunch.from_df('./data/', df, df_test, vocab=data_lm.train_ds.vocab, bs=64, text_cols=0, label_cols=1)

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')

learn.fit_one_cycle(1, 1e-2)


learn.predict("This was a great movie!!")

