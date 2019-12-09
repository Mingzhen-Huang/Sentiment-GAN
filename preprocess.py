from fastai import *
from fastai.text import *
import pandas as pd

poems_df = pd.read_json('./data/gutenberg-poetry.txt-v001.ndjson',lines=True)
poems_df['s'] = poems_df['s'].astype('str')
print(poems_df)

data_lm = (TextList.from_df(poems_df.loc[:50000,:],cols='s')
	           .split_by_rand_pct(0.1)
	           .label_for_lm()
	           .databunch(bs=64))

data_lm.save('./data/poems_tmp')
