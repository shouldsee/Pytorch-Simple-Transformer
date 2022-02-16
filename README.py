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


20220216 Notes

After many rounds of testing a generative mixture RNN model that performs different y=Wx+b
according to its partition within the phase space, I found it is impossible to train a
model by sampling from the generative model and then maximise the log-likelihood of the
emission. After some thoughts and comparison with classical HMM and RNN, I realised that
this phenomena is very intuitive -- I was trying to generate all different kinds of sentences
with the same latent variable. This way, the model is unable to learn the correspondence
between the latent variable and the emission. Consider a simple example, the MCMC will
sample variable from component A and component B, but both data points are connected to
the observed y in the emission space, thus all compoents are feeling the same pull
force from the observed data, thus unable to differentiate into different distributions,
(which is the complete opposite of mode collapsing problem,, where the latent variable
is the exact copy of the observed variables, instead of completely decorrelated ).
Thus, in order for the prob dist to differentiate, one should not sample from the
generative model alone, but needs to also bias the sampling given the observation,
commonly by sampling the posterior prob dist given the emission, and then calculating
the log-likelihood using the posterior latent variable samples. This scheme is commonly
known as the Expectation-Maximisation algorithm, where in the E step
one estimate the latent variable given the model params, and in the M-step one estimate
the model parameters given the latent variables.

From this angle, the RNN-training in neural language models can be understood not just
as a series of differetiable transformations, but also an implicit EM model, where
z(t+1) = W z(t) + w(t) is basically calculating a MLE point estimate of the latent variable
given z(t) and w(t) are mean vectors of gaussian of the same dispersion. And the
softmax emission layer p(w(t+1)) = f(z(t+1)) is calculating the log-likelihood given the
latent variable.

In practice, fitting with point MLE of latent variables could get stuck if there are
discrete variables, where usage of MCMC could mitigate the situation by providing
a more diversed sample.

Given this line in mind, we continue to ask what is the statitical nature of word2vec
models like CBOW and Skip-Gram. In those models each word is equipped with a forward
vector and a backward vector, that could also be seen as MLE estimate of the latent variable
given the current model, and inferring the posterior is considerably simple - just
lookup the table and take the sum/average, which is what you do to estimate the mean vector
of a gaussian (doge :D). This reflects the implicit gaussian assumption in many neural networks

How this understanding could aid the building of neural networks? I am not quite sure yet.
1-gram model is surely flawed because it could not even handle the existence of space and
every information is lost once you got a space in your tokens. Self-attention is intriguing
but I am not yet comfortable with the idea of a weight matrix to generate Q,K,V. In fact,
I believe big weight matrices are difficult to train, because it could be contracting or enlarging
prob dist if not properly normalised. Instead, we could use small rotations matrices and lower
dimensional affine transformations instead. But before a detailed discussion on this matter,
I want to understand first what exactly is a transformer network if not self-attened.
Consider a shallow transformer where the Q vectors are fixed and the K,V are trained for
each word, a series of Q will be finding the most approriate neighbors and extracting their
values. This is very similar to the word2vec model, where a q(t-1) is compared to k(t) and
choice of w(t) gives rise to the v(t), which is then assigned to q(t) and starts the next cycle.

q(t) + Array of k(t+1) -> Selecting the best matching v(t+1) -> assigns to q(t+1) -> ...

The most important thing here is that discrete transition w(t) -> w(t+1) is modelled with the q and k vector.
Where one can model interaction of thousands of words with each other implicitly and efficiently,
instead of having to work with the discrete probabilities. In fact, this means existence of
tokens are still important, and we can think of the tokens are just meta-words. The generative process of
the model is not easy to write down..., consider the encoding process first, one is provided with
the k(i) and v(i) tokens and the network holds q(i+1) and k(i+1) tokens, which are then used to
generate and v(i+1). If we make q(i) and k(i) static, then the only dynamical part will be
v(i), which is essentially a static network. Thus, one needs to make k(i+1) dynamical as well,
which means k(i+1) and v(i+1) could just be different dims of the same vector. This process
is essentially the opposite of the 1-gram model, where one push back from a pool of w(i+1) with k(i+1),
then uses q(i) to to select the best matching word from the pool. Instead of selecting the word to
generate, one matches the word that is most possibly generated (aka producing a child versus
finding the child among the peers.). It's not so obvious but maybe a such model would be non-generative process
due to this inversion.

After one round of selection, one is done with a sorted representation, and needs to do further
computation to do useful stuff, we call this KV attention, where the words stores KV

(k(i),v(i)) + (q(i+1)) -> (k(i+1)=q(i+1),v(i+1)) + q(i+2)  -> (k(i+2),v(i+2))


Let's consider the opposite transformation by inverting the model, where instead of k(i), we treat the
sentence as q(i), and we have (Q attention), where the words stores Q

q(i) + (k(i),v(i)) -> q(i+1) + (k(i+1),v(i+1)) -> q(i+2)

This way, we are replacing the original sentence with a set of pre-defined meta tokens.

In fact, the naturally sampling direction would be take the v and construct the next q vector. And then
query a KV pool. In KV attention, one construct a KV pool and then scans it with q to get V, but then
needs to reconstruct the K from the V, which seems rather weird.

In Q attention, we consider the next layer to be the most possible tokens given the input, with those
tokens, one then select the most likely next rounds of tokens, basically a re-tokenization process that
is inherently sorting the words into implicit prototypes. One could think the Q attention is growing
vertical sentence on the given sentence. But in fact, this type of Q attention does not calculate
the dynamics between the nodes, where each node evolves independently of each other, and show a
static computation graph.

Thus to gain dynamical computation graph one still needs to use KV attention, which is inferring the
latent parent giving rise to the current emission, and excluding the non-children datapoints when
estimating the generating distribution.

This standard KV attention model is not enough since all it does is a implicit clustering on the
emitted vectors where a query is used to seed a gaussion MLE point iteratively. That's why I am
making a slight twitch to the model. Grabbing the vector is important for sure, but modelling
the interaction between the vectors is more important. And this is done by selecting compatible
vectors and then contracting the middle dimension.

Consider

'''
