import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

"""
Transformer Implementation By Chenrong Lu 2021
Some Layers Refer to The Annotated Transformer (Harvard NLP)
"""
# class SelfAttentionOld(nn.Module):
#     def __init__(self,embed_dim,d_k,d_v,mask=False):
#         super(SelfAttentionOld,self).__init__()
#         self.query_embed = nn.Linear(embed_dim,d_k)
#         self.key_embed = nn.Linear(embed_dim,d_k)
#         self.value_embed = nn.Linear(embed_dim,d_v)
#         self.d_k = d_k
#         self.mask = mask
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self,query_in,key_in,value_in):
#         query = self.query_embed (query_in)
#         key   = self.key_embed   (key_in)
#         value = self.value_embed (value_in)
#         # value = value_in
#
#         # value = value_in
#         # value = value_in
#         key_transposed = torch.transpose(key,1,2)
#         #Get attention weights
#         attention_weights = torch.matmul(query,key_transposed)  #(n_query,n_key)
#         attention_weights = attention_weights - 0.5 * torch.sum(torch.square(query),dim=2,keepdim=True)
#         attention_weights = attention_weights - 0.5 * torch.transpose(torch.sum(torch.square(key),dim=2,keepdim=True),1,2)
#         attention_weights = attention_weights/math.sqrt(self.d_k)
#         if(self.mask==True):
#             #REF : http://peterbloem.nl/blog/transformers
#             indices = torch.triu_indices(attention_weights.shape[1],attention_weights.shape[2], offset=1)
#             attention_weights[:, indices[0], indices[1]] = float('-inf')
#         # attention_weights = torch.abs(attention_weights)
#         attention_weights = F.softmax(attention_weights, dim=2)
#         # attention_weights = attention_weights - torch.transpose(attention_weights,1,2)
#         attention_weights = attention_weights - torch.mean(attention_weights,dim=2,keepdim=True)
#
#         # attention_weights
#         #Apply attention weights to value
#         attention_weighted_value = torch.matmul(attention_weights,value) #(n_query,n_key) matmul (n_key || n_query , d_v)
#         # attention_weighted_value = self.value_embed(attention_weighted_value)
#         attention_weighted_value = self.dropout(attention_weighted_value)
#         return attention_weighted_value



class StateTransfer(nn.Module):
    def __init__(self, embed_dim, d_ker, mask=False,CUDA=False,
    #is_value_embed=False,
    is_state_transfer = True,
    #return_attention=True
    ):
        #### Use Attention to move a location Vector
        super(StateTransfer,self).__init__()
        self.attention = GenericAttention(
            d_q = embed_dim,
            d_k = embed_dim,
            d_ker = d_ker,
            d_v = 1,
            d_o = 1,
            is_state_transfer = is_state_transfer,
            is_value_embed= False,
            return_attention = True,
            mask=mask,
            CUDA=CUDA
            ### Locatoin Vector is a scalar that can be soft maxed
            ###
              )
    def forward(self, q,k,v):
        return self.attention(q,k,v)

class SelfAttention(nn.Module):
    def __init__(self,embed_dim,d_ker,d_v,mask=False,CUDA=False,is_value_embed = True, ):
        super(SelfAttention,self).__init__()
        self.attention = GenericAttention(
            d_q = embed_dim,
            d_k = embed_dim,
            d_v = embed_dim,
            d_ker = d_ker,
            d_o = d_v,
            is_value_embed = is_value_embed)
    def forward(self, q,k,v):
        return self.attention(q,k,v)

class GenericAttention(nn.Module):
    def __init__(self, d_q, d_k, d_ker, d_v, d_o , is_value_embed, is_state_transfer = False, return_attention = False,mask=False,CUDA=False):

    # def __init__(self,embed_dim,d_k,d_v,mask=False,CUDA=False):
        super(GenericAttention,self).__init__()
        # self.embed_value = embed_value
        self.is_state_transfer = is_state_transfer
        self.is_value_embed = is_value_embed
        self.return_attention = return_attention
        # self.query_embed = nn.Linear(embed_dim,d_k,bias=False)
        # self.key_embed   = nn.Linear(embed_dim,d_k,bias=False)
        # self.value_embed = nn.Linear(embed_dim,d_v,bias=False)
        self.query_embed = nn.Linear(d_q, d_ker)
        self.key_embed   = nn.Linear(d_k, d_ker)
        self.value_embed = nn.Linear(d_v, d_o)
        self.d_k  = d_k
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

    def forward(self,query_in,key_in,value_in):

        # query_weight
        query = self.query_embed (query_in)
        # qwt   = torch.softmax(self.query_embed.weight,dim=0)
        # qwt   = qwt -torch.mean(qwt,dim=0,keepdim=True)
        # query = torch.matmul(query_in, qwt)

        key   = self.key_embed   (key_in)
        # kwt   = torch.softmax(self.key_embed.weight,dim=0)
        # kwt   = kwt -torch.mean(kwt,dim=0,keepdim=True)
        # key   = torch.matmul(key_in,kwt)

        value = self.value_embed (value_in) if self.is_value_embed else value_in
        # value =

        # query = to
        # query = query / (torch.mean(torch.abs(query),dim=2,keepdim=True))
        # key   = key   / (torch.mean(torch.abs(key),dim=2,keepdim=True))
        key_transposed = torch.transpose(key,1,2)
        #Get attention weights
        attention_weights = torch.matmul(query,key_transposed)  #(n_query,n_key)
        # print(attention_weights.shape)
        attention_weights = attention_weights - 0.5 * torch.sum(torch.square(query),dim=2,keepdim=True)
        attention_weights = attention_weights - 0.5 * torch.transpose(torch.sum(torch.square(key),dim=2,keepdim=True),1,2)
        attention_weights = attention_weights/math.sqrt(self.d_k)
        if(self.mask==True):
            for i in range(attention_weights.shape[1]):
                attention_weights[:,i,i] = float('-inf')
            # #REF : http://peterbloem.nl/blog/transformers
            # indices = torch.triu_indices(attention_weights.shape[1],attention_weights.shape[2], offset=1)
            # attention_weights[:, indices[0], indices[1]] = float('-inf')
        shape0 = attention_weights.shape
        if self.is_state_transfer:
            attention_weights = F.softmax(attention_weights,dim=1)
        else:
            attention_weights = F.softmax(attention_weights,dim=2)
        if self.return_attention:
            return attention_weights
        # attention_weights = attention_weights *0.
        # attention_weights = torch.exp(attention_weights)

        # attention_weights = F.softmax(attention_weights.reshape((shape0[0],-1)), dim=1)
        # attention_weights = attention_weights.reshape(shape0)
        # ss = attention_weights - torch.max(attention_weights,dim=(2,))
        # attention_weights = attention_weights - torch.mean(attention_weights,dim=2,keepdim=True)

        #Apply attention weights to value
        attention_weighted_value = torch.matmul(attention_weights,value) #(n_query,n_key) matmul (n_key || n_query , d_v)
        # attention_weighted_value = self.dropout(attention_weighted_value)
        return attention_weighted_value

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,d_k,d_v,num_heads,mask=False,CUDA=False,is_value_embed=True):
        super(MultiHeadAttention,self).__init__()
        self.attention_blocks = [
            SelfAttention(embed_dim,d_k,d_v,mask,is_value_embed=is_value_embed) for _ in range(num_heads)
        ]
        [setattr(self,'attention_block_%d'%xi,xxx) for xi,xxx in enumerate(self.attention_blocks)]
        self.norm = LayerNorm(embed_dim)
        self.CUDA = CUDA
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
    def forward(self,query,key,value,residual_x):
        # residual_x = 0.
        attention_out = torch.tensor([],requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat((attention_out,attention(query,key,value)),dim=2)
        add_and_norm = self.norm(attention_out + residual_x)
        return add_and_norm


class LayerNorm(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        div =  (std + self.eps) + self.shift

        return self.scale * (x - mean) / div
        #/(div)
class PositionWiseFeedForward(nn.Module):
    def __init__(self,embed_dim,output_dim):
        super(PositionWiseFeedForward,self).__init__()
        self.l1 = nn.Linear(embed_dim,output_dim)
        self.RELU = nn.ReLU()
        self.l2 = nn.Linear(output_dim,embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self,x,residual_x):
        x = torch.max(torch.zeros(x.shape),self.l1(x))
        x = self.RELU(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.norm(x + residual_x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,is_value_embed=True,mask=False,CUDA=False):
        super(TransformerBlock,self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim,embed_dim//num_heads,embed_dim//num_heads,num_heads,mask,CUDA=CUDA,
        is_value_embed=is_value_embed)
        self.feed_forward = PositionWiseFeedForward(embed_dim,embed_dim)
    def forward(self,query,key,value,residual_x):
        attention_out = self.multi_head_attention(query,key,value,residual_x)
        # return attention_out
        feed_forward_out = self.feed_forward(attention_out,attention_out)
        return feed_forward_out

class VocabLogits(nn.Module):
    def __init__(self,embed_dim,logit_dim,CUDA=False):
        super(VocabLogits,self).__init__()
        self.linear = nn.Linear(embed_dim,logit_dim)
    def forward(self,x):
        return F.log_softmax(self.linear(x),dim=-1)

class Embeddings(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"
    def __init__(self,vocab_length,embed_dim,CUDA=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_length, embed_dim)
        # self.pos_encode = PositionalEncoding(embed_dim,CUDA=CUDA)
        self.embed_dim = embed_dim
    def forward(self, x):
        embed = (self.lut(x) * math.sqrt(self.embed_dim))
        return embed
        # +self.pos_encode(embed)

class PositionalEncoding(nn.Module):
    "Modified From Annotated Transformer (HarvardNLP)"
    def __init__(self, embed_dim,max_len=5000,CUDA=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term_even = torch.pow(10000.0,torch.arange(0, embed_dim, 2,dtype=torch.float32)/ embed_dim)
        div_term_odd = torch.pow(10000.0,torch.arange(1, embed_dim, 2,dtype=torch.float32)/ embed_dim)

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        pe = pe.unsqueeze(0)
        self.pe = pe
        # if(CUDA==True):
        #     pe.type(torch.cuda.FloatTensor)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x
