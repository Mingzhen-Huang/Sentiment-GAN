from fastai import *
from fastai.text import *

path = Path('./data')
data_lm = load_data(path, 'poems_tmp')
learn = language_model_learner(data_lm, arch=AWD_LSTM)
learn.load('poems_gan_gumbel')
print(learn.predict("Red",n_words=50))


