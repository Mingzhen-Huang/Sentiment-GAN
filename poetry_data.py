from fastai import *
from fastai.text import *
import pandas as pd


# with open('newpoetry.txt','r') as f:
#     text = f.readlines()
#     new_text = []
#     count = 0
#     for i in text:
#         new_text.append(i)
#         count+=1
#         if count == 14:
#         	count=0
#         	new_text.append('\n')
#         	new_text.append('\n')
#     with open('newpoetrys.txt','w') as nf:
#         nf.writelines(new_text)

# poems_df = pd.read_fwf('newpoetry.txt')

# poems_df['s'] = poems_df['s'].astype('str')
# print(poems_df)

# data_lm = (TextList.from_df(poems_df.loc[:50000],cols='s')
# 	           .split_by_rand_pct(0.1)
# 	           .label_for_lm()
# 	           .databunch(bs=64))

# data_lm.save('./data/poems_tmp')


df = pd.read_csv('kaggle_poem_dataset.csv')['Content']
# path = Path('./')
# bs = 64
# poems = (path/'newpoetrys.txt').open().read().split('\n\n')
# poems_df = pd.DataFrame(poems)
# tokenizer = Tokenizer(SpacyTokenizer, 'en')
# processor = [TokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(min_freq=1,max_vocab=60000)]
# data_lm = (TextList.from_df(poems_df,path,cols=0,processor=processor)
#             .split_by_rand_pct(0.1)
#             .label_for_lm()           
#             .databunch(bs=bs))
# data_lm.save('./data/poems_tmp')
