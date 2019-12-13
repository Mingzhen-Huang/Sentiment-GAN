# Sentiment-GAN
### Code base

-   Pretrained models and training ideas are from Fast.ai document: <https://docs.fast.ai/text.html>
-   Referred code of Creative GANs for generating poems, lyrics, and metaphors: <https://github.com/Machine-Learning-Tokyo/Poetry-GAN>

### Modified files

Detail:

train process part of in `train.py`  are inspired by code of Creative GANs for generating poems, lyrics, and metaphors: <https://github.com/Machine-Learning-Tokyo/Poetry-GAN>, other part are written by ourselves.

`util.py` and `gan.py` files are based on some functions of code of Creative GANs for generating poems, lyrics, and metaphors: <https://github.com/Machine-Learning-Tokyo/Poetry-GAN>.

`sentiment_discriminator.py` file are slightly based on Fast.ai’s tutorial <https://docs.fast.ai/text.html>.

`interface.py` file are written by ourselves.

`sentiment_loss.py` file are written by ourselves.

`preprocess.py` file are written by ourselves.

`poetry_data.py` file are written by ourselves.

### How to use

Please run the program step by step:

1.  Train the language model of poetry:

Can simply run:

```
python train.py --train_lm
```

Or can modify some hyperparameters:

```
python train.py --train_lm
		--lm_lr # learning rate, type=float, default=1e-3
  		--lm_epoch # training epoch, type=int, default=8
    	        --pretrain # 
```

2.  Train the sentiment discriminator:

```
python sentiment_discriminator.py
```

3.  Train the sentiment-GAN:

Can simply run:

```
python train.py --train_gan
```

Or can modify some hyperparameters:

```
python train.py --train_gan
		--path PATH # path of data, type=str, default='./data'
                --gan_epoch # training epoch, type=int, default=8
                --negative # sentiment of poem, true for negative, false for positive, type=bool, default=True
                --optim # optimizer, 'adam' or 'sgd', type=str,  default='adam'
                --d_lr # learning rate of text discriminator, type=float,  default=1e-3
                --g_lr # learning rate of generator, type=float,  default=1e-3
```

4.  Use interface to generate poetry:

```
python interface.py --word # beginning words, type=str, default='Red'
		    --n_words # number of predictions, type=int, default=50
```



### Requirements

Can simply run the follow to meet the requirements:

```
pip install -r requirements.txt
```

Core requirements:

```
fastai==1.0.59
torch==1.3.1
```
