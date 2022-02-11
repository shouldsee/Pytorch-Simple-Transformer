'''
Motivation:

Attention is hot, but I struggle to find the relevant intepretation
from a background in Bioinformatics and Statistical learning. More
specifically, I feel unable to understand what exactly is the intuition
behind the very core of attention:
$ a_{ij} = Attention(Q,K,V) = Softmax( Q_i \cdot K_j^T ) V_j $

After spending my master degree and 2 following years to apply mixture models
to RNA-Seq datasets, I recognised softmax function as an old friend from mixture model,
and from my perspective, attention is a really beautiful application of mixture
modelling to neural networks. Unlike other approaches asking ANN to predict
parameters of probability density distributions and then mixing them together,
attention network makes themselves a massive mixture model, thus permitting
a theoretical intepretation along this line.

In this blog, I will try to provide an experimental analysis of the attention
mechanism using neural machine translation as an example. Different model
architecture will be tested on English-German dataset, and reasons on why
one model out-performs another will be discussed. The models to be compared
are:


- Simple RNN Encoder-Decoder Model
- Mixture RNN Encoder-Decoder Model
- Second-order Mixture RNN Encoder-Decoder Model
- Transformer RNN Encoder-Decoder Model
'''
