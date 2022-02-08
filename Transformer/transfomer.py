import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Transformer.sub_layers import (MultiHeadAttention, PositionalEncoding,
LayerNorm,
SelfAttention,
GenericAttention,
StateTransfer,

                        PositionWiseFeedForward, TransformerBlock, VocabLogits,Embeddings)

import math
def ceil_log2(l):
    embed = math.ceil(math.log2(l))
    return int(embed)
# import numpy as np
def BinaryPositionEncoding(shape,dim):
    l = shape[dim]
    embed = ceil_log2(l)
    # l = torch
    x = torch.zeros(list(shape)+[embed])
    shape2 = [ int(xx) if ixx==dim else 1  for ixx,xx in enumerate(x.shape)]
    code = torch.range(0,l-1).reshape(shape2)
    vs = []
    for vv in range(embed):
        vs.append(code % 2)
        code = code //2
    vs = torch.cat(vs,dim=-1)
    vs = vs-0.5
    x = x+vs
    return x


class Encoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_blocks,CUDA=False):
        super(Encoder, self).__init__()
        self.transformer_blocks = [
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA) for _ in range(num_blocks)
        ]
        [setattr(self,'transformer_block_%d'%xi,xxx) for xi,xxx in enumerate(self.transformer_blocks)]
        self.positional_encoding = PositionalEncoding(embed_dim)
    def forward(self, x):
        # x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            # block0 = self.transformer_blocks[0]
            block0 = block
            x = block0(x,x,x,x)
        return x


class OutputRotator(nn.Module):
    def __init__(self,embed_dim,):
        pass
import pandas as pd

class PositionalDecoder(nn.Module):
    def __init__(self, embed_dim, l, dout):

        super(PositionalDecoder,self).__init__()
        # bpe = BinaryPositionEncoding( x,-1)
        self.l = l
        self.pe_len = pelen = ceil_log2(l)
        self.attention = GenericAttention(
            d_q   = pelen,
            d_k   = embed_dim,
            d_v   = embed_dim,
            d_ker = embed_dim,
            d_o   = dout
            )
        self.norm = LayerNorm(embed_dim)

        # self.transformer_block = GenericAttention( self.pe_len, embed_dim, dout)

    def forward(self,x):
        bpe = BinaryPositionEncoding([len(x), self.l],1)[:,:x.shape[1]]
        y = self.attention( bpe, x, x)
        y = self.norm(y)
        return y

class Encoder2(nn.Module):
    def __init__(self,embed_dim,num_heads,num_blocks,CUDA=False):
        super(Encoder2, self).__init__()
        self.transformer_blocks = [

            (
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA),
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA)
            )
             for _ in range(num_blocks)
        ]
        # self.state_transfer_block = StateTransfer(embed_dim, embed_dim, mask=False,CUDA=CUDA)
        [setattr(self,'transformer_block0_%d'%xi,xxx[0]) for xi,xxx in enumerate(self.transformer_blocks)]
        [setattr(self,'transformer_block1_%d'%xi,xxx[1]) for xi,xxx in enumerate(self.transformer_blocks)]
        self.num_blocks = num_blocks
        self.positional_encoding = PositionalEncoding(embed_dim)
        # self.norm = LayerNorm(embed_dim)
    def forward(self, x):
        # x = self.positional_encoding(x)
        # print(y.shape)
        # for i in range(y.shape[1]):
        #     y[:,i,i] = 1
        # norm = transformer_blocks
        # norm = self.transformer_blocks[0][0].multi_head_attention.norm
        # y = norm(y)
        # y = torch.zeros((x.shape[0],10,10))        z = torch.

        # lv = torch.zeros(list(x.shape)[:2]+[1])
        # lv[:,0,:]=1
        for i in range(self.num_blocks):
            ys = []
            # for _,block1 in self.transformer_blocks:
            #     y = torch.zeros((x.shape[0], 1, x.shape[2]))
            #     y = block1(y,x,x,y)
            #     ys.append(y)
            block0,block1 = self.transformer_blocks[i]
            # y = torch.cat(ys,dim=1)
            # if i+1==self.num_blocks:
            #     x = block0(x,y,y,x)
            # else:
            #     x = block0(x,y,y,x)

            x = block0(x,x,x,x)
        return x

class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_blocks,vocab_size,CUDA=False):
        super(Decoder, self).__init__()
        # self.multi_head_attention = MultiHeadAttention(embed_dim,embed_dim//num_heads,embed_dim//num_heads,num_heads,mask=False,CUDA=CUDA)
        self.transformer_blocks = xx = [
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA) for _ in range(num_blocks)
        ]
        [setattr(self,'transformer_block_%d'%xi,xxx) for xi,xxx in enumerate(self.transformer_blocks)]
        self.vocab_logits = VocabLogits(embed_dim,vocab_size)
    def forward(self, encoder_outs,x):
        for block in self.transformer_blocks:
            z = torch.cat([encoder_outs,x],dim=1)
            x = self.transformer_blocks[0](query=x[:,-1:,:],
                                                                key=z,
                                                                value=z,
                                                                residual_x=x[:,-1:,:])
            # x = block(query=output_seq_attention_out,
            #           key=encoder_outs,
            #           value=encoder_outs,
            #           residual_x=output_seq_attention_out)
        return self.vocab_logits(x)


class DecoderOld(nn.Module):
    def __init__(self,embed_dim,num_heads,num_blocks,vocab_size,CUDA=False):
        super(Decoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim,embed_dim//num_heads,embed_dim//num_heads,num_heads,mask=False,CUDA=CUDA)
        self.transformer_blocks = xx = [
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA) for _ in range(num_blocks)
        ]
        [setattr(self,'transformer_block_%d'%xi,xxx) for xi,xxx in enumerate(self.transformer_blocks)]
        self.vocab_logits = VocabLogits(embed_dim,vocab_size)
    def forward(self, encoder_outs,x):
        for block in self.transformer_blocks:
            output_seq_attention_out = self.multi_head_attention(query=x[:,-1:,:],
                                                                key=x[:,:-1],
                                                                value=x[:,:-1],
                                                                residual_x=x[:,-1:,:])
            x = block(query=output_seq_attention_out,
                      key=encoder_outs,
                      value=encoder_outs,
                      residual_x=output_seq_attention_out)
        return self.vocab_logits(x)

#
# class TransformerTranslator(nn.Module):
#     def __init__(self,embed_dim,num_blocks,num_heads,vocab_size,CUDA=False):
#         super(TransformerTranslator,self).__init__()
#         self.embedding = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
#         self.encoder = Encoder(embed_dim,num_heads,num_blocks,CUDA=CUDA)
#         self.decoder = Decoder(embed_dim,num_heads,num_blocks,vocab_size,CUDA=CUDA)
#         self.decoder2 = Encoder(embed_dim,num_heads,num_blocks,CUDA=CUDA)
#         self.encoded = False
#         self.device = torch.device('cuda:0' if CUDA else 'cpu')
#     def encode(self,input_sequence):
#         embedding = self.embedding(input_sequence).to(self.device)
#         self.encode_out = self.encoder(embedding)
#         self.encoded = True
#     def forward(self,output_sequence):
#         if(self.encoded==False):
#             print("ERROR::TransformerTranslator:: MUST ENCODE FIRST.")
#             return output_sequence
#         else:
#             embedding = self.embedding(output_sequence)
#             return self.decoder(self.encode_out,embedding)


class TransformerTranslatorFeng(nn.Module):
    def __init__(self,embed_dim,num_blocks,num_heads,vocab_size,
    output_vocab_size,max_context_length,CUDA=False):
        super(TransformerTranslatorFeng,self).__init__()
        self.embedding = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding2 = Embeddings(output_vocab_size,embed_dim,CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)
        self.encoder = Encoder2(embed_dim,num_heads,num_blocks,CUDA=CUDA)
        self.decoder = Decoder(embed_dim,num_heads,num_blocks,output_vocab_size,CUDA=CUDA)
        # self.decoder2 = Encoder2(embed_dim,num_heads,num_blocks,CUDA=CUDA)
        self.pdec     = PositionalDecoder(embed_dim, max_context_length, embed_dim)
        # self.embedding_pe_enc = nn.Linear(embed_dim+self.pdec.pe_len,embed_dim)
        self.encoded = False
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encode_out = None
    def encode(self,input_sequence):
        x=  embedding = self.embedding(input_sequence).to(self.device)[:,:,:]

        x = x[:,:,:-self.pdec.pe_len]
        bpe = BinaryPositionEncoding([len(x), self.pdec.l],1)[:,:x.shape[1]]
        x = torch.cat([x,bpe],dim=2)

        # x = self.embedding_pe_enc(x)

        self.encode_out = self.encoder(x)
        self.encoded = True
    def forward(self, output_sequence):
        if(self.encoded==False):
            print("ERROR::TransformerTranslator:: MUST ENCODE FIRST.")
            return output_sequence
        else:
            x = self.embedding2(output_sequence)

            x = x[:,:,:-self.pdec.pe_len]
            bpe = BinaryPositionEncoding([len(x), self.pdec.l],1)[:,:x.shape[1]]
            x = torch.cat([x,bpe],dim=2)

            x =  self.decoder(self.encode_out, x)
            # x =  self.pdec(x)
            return (x)

            # embedding = self.embedding2(output_sequence)
            # return self.decoder(self.encode_out,embedding)
