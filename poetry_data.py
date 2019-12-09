from fastai import *
from fastai.text import *
import pandas as pd


# with open('data/poetry.txt','r') as f:
#     text = f.readlines()
#     new_text = []
#     for i in text:
#         if len(i) > 8:
#             new_text.append(i)
#     with open('newpoetry.txt','w') as newf:
#         newf.writelines(new_text)

poems_df = pd.read_fwf('newpoetry.txt')

poems_df['s'] = poems_df['s'].astype('str')


data_lm = (TextList.from_df(poems_df.loc[:50000],cols='s')
	           .split_by_rand_pct(0.1)
	           .label_for_lm()
	           .databunch(bs=64))

data_lm.save('./data/poems_tmp')