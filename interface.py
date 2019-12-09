from fastai import *
from fastai.text import *
import argparse

path = Path('./data')
data_lm = load_data(path, 'poems_tmp')
learn = language_model_learner(data_lm, arch=AWD_LSTM)
learn.load('poems_gan_gumbel')

parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--word', type=str, help="Beginning words", default='Red')
parser.add_argument('--n_words', type=int, help="Number of predictions", default=50)

args = parser.parse_args()


print(learn.predict(args.word, n_words=args.n_words))



