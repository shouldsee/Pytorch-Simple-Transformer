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
            d_o   = dout,
            is_value_embed = True,
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
        n_past = 5
        self.embed_dim  =embed_dim
        self.n_past = n_past
        self.transformer_blocks = [

            (
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA),
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA)
            )
             for _ in range(num_blocks)
        ]
        self.state_transfer_block = StateTransfer(embed_dim, embed_dim, mask=True,CUDA=CUDA, is_state_transfer=False)
        self.attention_block_list = nn.ModuleList([SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 0,mask=False,CUDA=CUDA) for _ in range(num_blocks)])
        [setattr(self,'transformer_block0_%d'%xi,xxx[0]) for xi,xxx in enumerate(self.transformer_blocks)]
        [setattr(self,'transformer_block1_%d'%xi,xxx[1]) for xi,xxx in enumerate(self.transformer_blocks)]
        self.lin1 = nn.Linear(n_past*embed_dim,embed_dim)
        self.lin2 = nn.Linear((1+n_past)*embed_dim,(1+n_past)*embed_dim)
        self.init_hidden = nn.Linear(n_past,embed_dim)
        self.num_blocks = num_blocks
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.norm1 = LayerNorm(embed_dim)


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

        lv = torch.zeros(list(x.shape)[:2]+[1])                         # (batch, token, 1)
        lv[:,:,:]=1./(lv.shape[1])
        # lv[:,0,:]=1.#/(lv.shape[1])
        # lv
        # past_v = torch.zeros((x.shape[0], self.n_past, x.shape[2]))     # (batch, n_past, embed_dim)
        past_v = torch.cat([self.init_hidden.weight.T[None,:,:]*100 for _ in range(len(x))],dim=0)
        nk_vec = past_v[:,0:1,:]
        # for i in range(self.num_blocks):
        # print(lv.sum(dim=1)[0])
        # past
        # print((lv[0,:,0].cpu().detach().numpy()*100).astype(int))
        # print(lv.shape)
        # for i in range(3):
        # print()
        # for block0,block1 in self.transformer_blocks:
        for i in range(5):
            ys = []
            # for _,block1 in self.transformer_blocks:
            #     y = torch.zeros((x.shape[0], 1, x.shape[2]))
            #     y = block1(y,x,x,y)
            #     ys.append(y)
            block0,block1 = self.transformer_blocks[0]
            # y = torch.cat(ys,dim=1)
            # if i+1==self.num_blocks:
            #     x = block0(x,y,y,x)
            # else:
            #     x = block0(x,y,y,x)
            # nk_vec =

            # key = self.lin1(past_v.reshape([len(x),1,self.n_past*self.embed_dim]))

            v_curr = torch.matmul(torch.transpose(lv,1,2), x)
            # past_v_extra = torch.cat([v_curr,past_v[:,:-1], ], dim=1)
            past_v_extra = torch.cat([v_curr, nk_vec, past_v, ], dim=1)
            # print(past_v_extra[0][:,0].round())



            # print(lv.sum(dim=1)[0])
            # print((lv1[0,:,0].cpu().detach().numpy()*100).astype(int))
            # [ print('{:02d}'.format(xx)if xx>0 else '  ',end=' ')
            # for xx in  (
            # lv[0,:,0].cpu().detach().numpy()*100).astype(int).tolist() ]
            # print()
            # dx = self.lin1(past_v.reshape([x.shape[0],1,self.n_p-0.2219,ast*self.embed_dim]))
            # dx =
            # key = self.lin2(past_v_extra.reshape((len(x),1,-1))).reshape((len(x),self.n_past+1,self.embed_dim))
            # past_v_extra = past_v_extra + key
            # past_v_extra = self.norm1(past_v_extra)
            layer = self.attention_block_list[0]
            # print(past_v_extra[0][:,0].round())
            past_v_extra =layer( past_v_extra,  past_v_extra, past_v_extra,)
            # print(past_v_extra[0][:,0].round())
            past_v_extra = self.norm1(past_v_extra)
            lv = self.state_transfer_block(past_v_extra[:,1:2,:],x,0.)
            lv = torch.transpose(lv,1,2)
            # lv = lv - torch.mean(lv,dim=1,keepdims=True)

            # past_v_extra)
            past_v = past_v_extra[:,2:,:]
            dx = past_v[:,0:1,:]
        # print(past_v_extra[0][:,0:5].round())
        # print()
        # print(past_v)
        # torch.cat(past_)
        return past_v_extra[:,2:]
        # return past_v

        # return x

class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_blocks,vocab_size,CUDA=False):
        super(Decoder, self).__init__()
        # self.multi_head_attention = MultiHeadAttention(embed_dim,embed_dim//num_heads,embed_dim//num_heads,num_heads,mask=False,CUDA=CUDA)
        self.transformer_blocks = xx = [
            TransformerBlock(embed_dim,num_heads,mask=False,CUDA=CUDA,is_value_embed=False) for _ in range(num_blocks)
        ]
        [setattr(self,'transformer_block_%d'%xi,xxx) for xi,xxx in enumerate(self.transformer_blocks)]
        self.vocab_logits = VocabLogits(embed_dim,vocab_size)
    def forward(self, encoder_outs,x):
        # print(z[0].cpu().detach().numpy().astype(int))
        z = torch.cat([encoder_outs,x],dim=1)
        vv = torch.zeros((len(x),1,x.shape[2]))
        vv[:,:,-1]=1

        for block in self.transformer_blocks[:-1]:
            # z = self.transformer_blocks[0]
            z = block(z,z,z,z)
        x = self.transformer_blocks[-1](query=vv,
                                                                key=z,
                                                                value=z,
                                                                residual_x= 0.
                                                                )
                                                                # residual_x=x[:,-1:,:])
            # print(x[0][:,:5])
            # x = block(query=output_seq_attention_out,
            #           key=encoder_outs,
            #           value=encoder_outs,
            #           residual_x=output_seq_attention_out)
        y = self.vocab_logits(x)
        # print(y.shape)
        # print(y.argmax(axis=2)[0])
        # print(y[0][0][y.argmax(axis=2)[0]])
        # .max(axis=2)[0])
        return y


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



class BasicRNN(nn.Module):
    def __init__(self, embed_dim, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(BasicRNN, self).__init__()
        self.hidden_dim = embed_dim
        self.embed_dim = embed_dim
        self.embedding = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)
        self.transfer = nn.Linear(embed_dim,embed_dim)
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False
        # self.decoder_transfer = nn.Linear(embed_dim*2,embed_dim,)
        self.decoder_transfer = nn.Linear(embed_dim, embed_dim,)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device)
        z = torch.zeros_like(x[:,0:1,:])
        for i in range(x.shape[1]):
            z = z + self.transfer(z + x[:,i:i+1,:])
            z = self.norm1(z)
        return z

    # def
    def forward(self, hidden_state, output_sequence):
        x = self.embedding_out(output_sequence).to(self.device)
        k = hidden_state
        z = torch.zeros([x.shape[0],1,x.shape[2]])
        for i in range(x.shape[1]):
            z = z + x[:,i:i+1,:]
            z = z + self.decoder_transfer(z)
            # z = self.decoder_transfer(torch.cat([z,k],dim=2)) + z
            z = self.norm2(z)
        y = self.vocab(z)
        return z,y

    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)

class MixtureRNN(nn.Module):
    '''
    Apply an enriched dynamics by splitting the phase space into different regions
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(MixtureRNN, self).__init__()
        self.embed_dim = embed_dim

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False

        self.encoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)

        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device)
        z = torch.zeros_like(x[:,0:1,:])
        mover = self.encoder_mover.weight
        mover = (z+1)*mover[None,:,:]
        for i in range(x.shape[1]):
            z = z + x[:,i:i+1,:]
            g  = torch.cat([z, mover,],dim=1)
            dg =  self.encoder_attention(g,g,g)
            z = z + dg[:,0:1]
            z = self.norm1(z)
        return z

    def forward(self, hidden_state, output_sequence):
        x = self.embedding_out(output_sequence).to(self.device)
        z = hidden_state
        norm = self.norm2
        att  = self.decoder_attention
        mover = self.decoder_mover.weight
        mover = (torch.zeros_like(z)+1)*mover[None,:,:]

        for i in range(x.shape[1]):
            z = z + x[:,i:i+1,:]
            g  = torch.cat([z, mover,],dim=1)
            dg = att(g,g,g)
            z = z + dg[:,0:1]
            z = norm(z)
        y = self.vocab(z)
        return z,y
    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)

class AnchorMixtureRNN(nn.Module):
    '''
    Apply an enriched dynamics by splitting the phase space into different regions
    Anchors value will be updated if particular phase space is visited, to store memory
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(AnchorMixtureRNN, self).__init__()
        self.embed_dim = embed_dim

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False
        anchor_count = mixture_count
        self.encoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.encoder_anchor_key= nn.Linear(embed_dim,anchor_count,)
        self.encoder_anchor_attention = SelfAttention(
        embed_dim,embed_dim,embed_dim,is_value_embed = False, return_attention=True, is_state_transfer=True,mask=False,CUDA=CUDA)
        self.decoder_anchor_reader = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = True,mask=False,CUDA=CUDA)
        self.decoder_anchor_key= nn.Linear(embed_dim,anchor_count,)

        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

        # self.norm3 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device);
        xs = x.shape
        z = torch.zeros([xs[0],1,xs[2]])                 ### init at origin
        mover = self.encoder_mover.weight                ### load dynamics
        mover = (z+1)*mover[None,:,:]                    ### reshaping
        mover_att    = self.encoder_attention
        anchor_key   = self.encoder_anchor_key.weight
        anchor_key   = (z*0+1)*anchor_key[None,:,:]
        anchor_value = 0*anchor_key
        anchor_att   = self.encoder_anchor_attention
        norm = self.norm1
        anchor_norm  =self.norm1
        # anchor_norm  =self.norm3
        for i in range(x.shape[1]):
            # print((anchor_value[0,:,:4]*10).cpu().detach().numpy().astype('int'))
            # print()
            z  =  z + x[:,i:i+1,:]
            g  =  torch.cat([z, mover,],dim=1)
            dg =  mover_att(g,g,g)
            z  = z + dg[:,0:1]
            z  = norm(z)
            anchor_att_ret = anchor_att(anchor_key,z,z)
            # anchor_att_ret = anchor_att_ret - torch.mean(anchor_att_ret,dim=1,keepdims=True)
            # anchor_value  = anchor_value + anchor_att_ret *z
            # anchor_value  = norm(anchor_value)
            anchor_value  = (1-anchor_att_ret) * anchor_value + anchor_att_ret *z
            anchor_value  = anchor_norm(anchor_value)
            # , anchor_att_ret.sum(axis=1)[0])

        return (0*z,anchor_value)

    def forward(self, hidden_state, output_sequence):

        x = self.embedding_out(output_sequence).to(self.device)
        z, anchor_value = hidden_state
        norm = self.norm2
        att  = self.decoder_attention
        mover = self.decoder_mover.weight
        mover = (torch.zeros_like(z)+1)*mover[None,:,:]

        # anchor_key   = self.decoder_anchor_key.weight
        # anchor_key   = (z*0+1)*anchor_key[None,:,:]
        anchor_reader = self.decoder_anchor_reader

        for i in range(x.shape[1]):
            z  = z*1 + 1*x[:,i:i+1,:]
            # torch.cat([z,anchor_value])

            dg2 = anchor_reader(z, anchor_value, anchor_value)
            z = z + dg2
            z = norm(z)

        # y = self.decoder_anchor_key(z)
        # zoff = self.encoder_anchor_key.weight[None,0:1,:]

        # y = torch.matmul( z,self.embedding_out.lut.weight.T)
        # y =
        # y = F.log_softmax(y,dim=2)
        y = self.vocab(z)

        #### cut the hidden_state z here
        return (  z,anchor_value),y
    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)


# class RecurrentModel(nn.Module):
#     def step(self):
#         pass
#

class BeamAnchorMixtureRNN(nn.Module):
    '''
    Apply an enriched dynamics by splitting the phase space into different regions
    Anchors value will be updated if particular phase space is visited, to store memory
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(BeamAnchorMixtureRNN, self).__init__()
        self.embed_dim = embed_dim
        self.mixture_count = mixture_count

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False
        anchor_count = mixture_count
        self.encoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.encoder_anchor_key= nn.Linear(embed_dim,anchor_count,)
        self.encoder_anchor_attention = SelfAttention(
        embed_dim,embed_dim,embed_dim,is_value_embed = False, return_attention=True, is_state_transfer=True,mask=False,CUDA=CUDA)
        self.decoder_anchor_reader = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = True,mask=False,CUDA=CUDA)
        self.decoder_anchor_key= nn.Linear(embed_dim,anchor_count,)

        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_anchor_attention_log = SelfAttention(
        embed_dim,embed_dim,embed_dim,is_value_embed = False, return_attention=True, is_state_transfer=False,mask=False,CUDA=CUDA,
        return_log=True,
        )
        self.decoder_anchor_disp = nn.Linear(embed_dim,embed_dim)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.topk = 20
        # self.norm3 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device);
        xs = x.shape
        z = torch.zeros([xs[0],1,xs[2]])                 ### init at origin
        mover = self.encoder_mover.weight                ### load dynamics
        mover = (z+1)*mover[None,:,:]                    ### reshaping
        mover_att    = self.encoder_attention
        anchor_key   = self.encoder_anchor_key.weight
        anchor_key   = (z*0+1)*anchor_key[None,:,:]
        anchor_value = 0*anchor_key
        anchor_att   = self.encoder_anchor_attention
        norm = self.norm1
        anchor_norm  =self.norm1
        # anchor_norm  =self.norm3
        for i in range(x.shape[1]):
            # print((anchor_value[0,:,:4]*10).cpu().detach().numpy().astype('int'))
            # print()
            z  =  z + x[:,i:i+1,:]
            g  =  torch.cat([z, mover,],dim=1)
            dg =  mover_att(g,g,g)
            z  = z + dg[:,0:1]
            z  = norm(z)
            anchor_att_ret = anchor_att(anchor_key,z,z)
            # anchor_att_ret = anchor_att_ret - torch.mean(anchor_att_ret,dim=1,keepdims=True)
            # anchor_value  = anchor_value + anchor_att_ret *z
            # anchor_value  = norm(anchor_value)
            anchor_value  = (1-anchor_att_ret) * anchor_value + anchor_att_ret *z
            anchor_value  = anchor_norm(anchor_value)
            # , anchor_att_ret.sum(axis=1)[0])

        idxpointer = torch.tensor([],requires_grad=False,dtype=torch.long).to(self.device)
        znew    = torch.tensor([],requires_grad=True).to(self.device)
        z = z  + torch.zeros([xs[0], self.topk, xs[2]])
        anchor_att_ret = torch.zeros([xs[0],self.topk,1])

        # assert 0.
        return ( z , anchor_value, anchor_att_ret, idxpointer, 0*z[:,None,:,:])

    def step(self, hidden,x):
        x = self.embedding_out(x).to(self.device)
        z, anchor_value,anchor_att_ret,idxpointer,zs = hidden

        xs    = x.shape
        topk  = self.topk
        norm  = self.norm2
        att   = self.decoder_attention
        mover = self.decoder_mover.weight
        mover = torch.zeros([xs[0], 1, xs[2]])+mover[None,:,:]

        anchor_att_log   = self.decoder_anchor_attention_log
        anchor_reader    = self.decoder_anchor_reader

        anchor_att_ret = torch.zeros([xs[0],topk,1]) + anchor_att_ret
        # z = z + torch.zeros([xs[0], topk, xs[2]])
        nextk = self.mixture_count

        print = lambda *x:None

        z = z + torch.zeros([xs[0],topk,xs[2]])

        # achs = torch.cat([anchor_value[:,:-5,:],mover[:,:5,:]],dim=1)
        # anchor_value)




        # anchor_att_ret_new = anchor_att_log(mover,z,mover)
        # anchor_value_disp = self.decoder_anchor_disp(z)


        # achs = anchor_value
        achs = torch.cat([anchor_value[:,:-5,:],mover[:,:5,:]],dim=1)
        dg2 = anchor_reader(z, achs, achs)
        # anchor_value)

        anchor_att_ret_new = anchor_att_log(achs,z,achs)
        anchor_value_disp = self.decoder_anchor_disp(achs)
        # anchor_value_disp = att(z,achs,achs)

        if 0:
            dg2 = anchor_reader(z, anchor_value, anchor_value)
            anchor_att_ret_new = anchor_att_log(anchor_value,z,anchor_value)
            anchor_value_disp = self.decoder_anchor_disp(anchor_value)
        # anchor_value_disp = self.decoder_anchor_disp.weight[None,:,:]
        # achs)
        # anchor_value_disp = att(mover,z,z)
        # achs)

        # print(anchor_att_ret_new.exp().sum(dim=2))
        # print(anchor_att_ret_new.T.shape)
        # print(anchor_value_disp.shape)
        # print(anchor_att_ret.shape)
        # print(anchor_att_ret_new.shape)
        print('[z]',z.shape)
        # print(dg2.shape)
        anchor_value_cross = torch.transpose(anchor_att_ret_new[:,:,:],1,2) + anchor_att_ret
        # print(anchor_value_cross.shape)
        print(anchor_att_ret[0])
        print(anchor_att_ret_new[0])

        val,idx= anchor_value_cross.reshape((x.size(0),-1)).topk(topk,dim=1)

        znew  = z[:,:,None,:] + anchor_value_disp[:,None,:,:]
        znew  = znew.reshape((x.size(0),-1, self.embed_dim))

        znew = znew[torch.arange(z.size(0))[:,None]+torch.zeros_like(idx), idx,:]
        anchor_att_ret =  val[:,:,None]
        anchor_att_ret = anchor_att_ret - anchor_att_ret.max(dim=1,keepdims=True)[0]
        anchor_att_ret = anchor_att_ret

        z = znew
        idxpointer = torch.cat([ idxpointer, idx[:,None,:] ],dim=1)
        # fpointer =
        # print('[MAX]',bpointer[0].max())
        # print('[MAX]',idx[0].max())
        # print(anchor_value_cross[0])
        # zs = torch.cat([zs,z[:,None,:,:]],dim=1)
        zs = torch.cat([zs,z[:,None,:,:]],dim=1)

        # print(anchor_att_ret_new[0][:5,:5])
        # print(bpointer.max())
        print(idxpointer.shape)
        print(zs.shape)
        print()
        # print(anchor_att_ret[0])
        # print(z[0,:,0])

        z = z + 0*dg2
        z = norm(z)
        y_right = self.vocab(z[:,0:1,:])
        return (z, anchor_value,anchor_att_ret,idxpointer,zs),y_right

    def forward(self, hidden_state, output_sequence):
        z, anchor_value, anchor_att_ret, bpointer,zs = hidden_state

        y = self.vocab(z)[:,0:1,:]
        for i in range(x.shape[1]):
            hidden_state,y = self.step(hidden_state, x[:,i:i+1])

        #### cut the hidden_state z here
        return ( z,anchor_value, anchor_att_ret, bpointer,zs),y

import math


def decode_with_target(model,hidden, batch_target,is_greedy,print_debug=0):
    g = batch_target.shape
    self = model
    if isinstance(model,BeamAnchorMixtureRNN):
        '''Sample the top-k most likely chains using
        a beam search heuristic
        '''
        logps = torch.tensor([],requires_grad=True).to(self.device)
        for i in range(g[1]-1):
            ( z,anchor_value, anchor_att_ret, idxpointer,zs )  = hidden
            logps = torch.cat([logps,anchor_att_ret[:,None,:,0]],dim=1)
            hidden, yf = self.step(hidden, batch_target[:,i-1:i])

        ( z,anchor_value, anchor_att_ret, idxpointer,zs )  = hidden
        logps = torch.cat([logps,anchor_att_ret[:,None,:,0]],dim=1)

        nextk = self.mixture_count
        bpointer = idxpointer//nextk
        fpointer = idxpointer%nextk
        # print(bpointer.shape,zs.shape,batch_target.shape)
        topk = 20
        bp = torch.arange(topk)[None,None,:] + torch.zeros_like(idxpointer[:,0:1,:topk])
        pointers = bp * nextk
        for i in range(g[1]-1 -1,0 -1,-1):
            # print(i,g[1],bpointer.size(1))
            bpnew = torch.gather(idxpointer[:,i:i+1,:],dim=2,index=bp)
            # bpnew = bpnew // nextk
            pointers   = torch.cat([bpnew,pointers],dim=1)
            bp    = bpnew // nextk
        # print(pointers[0])


        pointers_back = pointers//nextk
        # print(pointers_back[0,-1])
        zsel     = torch.cat([torch.gather(zs[:,:,:,i:i+1],dim=2,index=pointers_back[:,:,:,None]) for i in range(zs.size(3))], dim=3)
        logpsel  = torch.gather(logps, 2, pointers_back)
        logpsel  = logpsel.sum(dim=1,keepdims=True)
        psel     = logpsel
        psel     = F.log_softmax(logpsel,dim=2)
        # - math.log(psel.size(2))
        out_prob_bychain = self.vocab(zsel)
        # psel  = logpsel - math.log(out_prob_bychain.size(3))
        # psel  = logpsel / (out_prob_bychain.size(3)) /
        # print(psel.shape,out_prob_bychain.shape)
        out_prob = torch.sum(torch.exp(psel[:,:,:,None]+ out_prob_bychain),dim=2)
        # out_prob = torch.sum(torch.exp(out_prob_bychain),dim=2)
        if print_debug:

            print(psel.shape,out_prob_bychain.shape)
            print(out_prob[0][5].sum(-1))
            print(pointers_back[0,-31:,:])
            print(torch.exp(psel[:,:,:,None]+out_prob_bychain)[0,1].sum(-1))
        # out_prob = torch.sum(torch.exp( out_prob_bychain),dim=2)
        # out_prob = torch
        eps=  1E-6
        # out_prob = F.softmax(psel[:,:,:,None]+ out_prob_bychain,dim=2)
        out_prob = torch.log(eps+ out_prob)
        # out_prob = torch.log_softmax(out_prob,dim=-1)
        out_prob = out_prob[:,:]
        # print(out_prob[0,0].sum(-1))
        model.pointers = pointers
        return out_prob


    else:
        x = torch.zeros( [g[0],g[1],],dtype=torch.long ).to(model.device)
        all_outs = torch.tensor([],requires_grad=True).to(model.device)
        for i in range(batch_target.shape[1]):
            xx = torch.zeros( [g[0],g[1], ],dtype=torch.long ).to(model.device)

            if is_greedy:
                hidden,out = model.step(hidden,  x[:,i-1:i])
                y = out.argmax(axis=-1)
                xx[:,i:i+1] = y
                x = x+xx
            else:
                hidden,out = model.step(hidden, batch_target[:,i-1:i])
            all_outs = torch.cat((all_outs,out),dim=1)
        return all_outs

class AnchorOnlyMixtureRNN(nn.Module):
    '''
    Apply an enriched dynamics by splitting the phase space into different regions
    Anchors value will be updated if particular phase space is visited, to store memory
    The original RNN dynamics is removed
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(AnchorOnlyMixtureRNN, self).__init__()
        self.embed_dim = embed_dim

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False
        anchor_count = mixture_count
        self.encoder_mover = nn.Linear(embed_dim,mixture_count,) ## [DEL]
        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.encoder_anchor_key= nn.Linear(embed_dim,anchor_count,)
        self.encoder_anchor_attention = SelfAttention(
        embed_dim,embed_dim,embed_dim,is_value_embed = False, return_attention=True, is_state_transfer=True,mask=False,CUDA=CUDA)
        self.decoder_anchor_reader = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = True,mask=False,CUDA=CUDA)
        self.decoder_anchor_key= nn.Linear(embed_dim,anchor_count,)

        self.decoder_anchor_attention = SelfAttention(
        embed_dim,embed_dim,embed_dim,is_value_embed = False, return_attention=True, is_state_transfer=True,mask=False,CUDA=CUDA)


        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)

        # self.norm3 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device);
        xs = x.shape
        z = torch.zeros([xs[0],1,xs[2]])                 ### init at origin

        anchor_key   = self.encoder_anchor_key.weight
        anchor_key   = (z*0+1)*anchor_key[None,:,:]
        anchor_value = 0*anchor_key
        anchor_att   = self.encoder_anchor_attention
        norm = self.norm1
        anchor_norm  =self.norm1
        mover_att    = self.encoder_attention
        # anchor_norm  =self.norm3
        for i in range(x.shape[1]):
            # print((anchor_value[0,:,:4]*10).cpu().detach().numpy().astype('int'))
            # print()
            z  =  z + x[:,i:i+1,:]
            z  = norm(z)
            anchor_att_ret = anchor_att(anchor_key,z,z)
            # anchor_att_ret = anchor_att_ret - torch.mean(anchor_att_ret,dim=1,keepdims=True)
            # anchor_value  = anchor_value + anchor_att_ret *z
            # anchor_value  = norm(anchor_value)
            anchor_value  = (1-anchor_att_ret) * anchor_value + anchor_att_ret *z
            anchor_value  = anchor_norm(anchor_value)
            # anchor_value  = anchor_value + mover_att(anchor_value,anchor_value,anchor_value)
            # anchor_value  = anchor_norm(anchor_value)
            # , anchor_att_ret.sum(axis=1)[0])

        return (z,anchor_value,0.)

    def forward(self, hidden_state, output_sequence):

        x = self.embedding_out(output_sequence).to(self.device)
        z, anchor_value,anchor_value_ximg = hidden_state
        norm = self.norm2
        att  = self.decoder_attention
        mover = self.decoder_mover.weight
        mover = (torch.zeros_like(z)+1)*mover[None,:,:]
        # anchor_key   = self.decoder_anchor_key.weight
        # anchor_key   = (z*0+1)*anchor_key[None,:,:]
        anchor_reader = self.decoder_anchor_reader

        anchor_att   = self.decoder_anchor_attention
        anchor_key_ximg   = self.decoder_anchor_key.weight
        anchor_key_ximg   = (z*0+1)*anchor_key_ximg[None,:,:]
        anchor_value_ximg = anchor_value_ximg + 0*anchor_key_ximg
        anchor_norm  =self.norm3

        for i in range(x.shape[1]):

            ########## we need to estimate the
            xc = 1*x[:,i:i+1,:]

            # z =  1* 0.333 * z + 0*0.333 *xc + 1*0.333 * ximg
            # z  = 0.3 * norm(z) + 0.7*  norm(xc)

            dg2 = anchor_reader(z, anchor_value, anchor_value)
            ### anchor is a read-only memory here and shall not be varied

            ximg = anchor_reader(z, anchor_key_ximg, anchor_value_ximg)

            anchor_att_ret     = anchor_att(anchor_key_ximg,z,z)
            anchor_value_ximg  = (1-anchor_att_ret) * anchor_value_ximg + anchor_att_ret *z
            anchor_value_ximg  = anchor_norm(anchor_value_ximg)


            # g  = torch.cat([z, mover,],dim=1)
            # dg  = att(g[:,0:1],g,g)

            z = z + dg2 + 0*ximg #+ 0.*dg
            z = norm(z)

        y = self.vocab(z)
        return (z,anchor_value,anchor_value_ximg),y

    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)


class DynamicMixtureRNN(nn.Module):
    '''
    E50 T84 V90
    no better than SOMRNN
    Apply an enriched dynamics by splitting the phase space into different regions
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(DynamicMixtureRNN, self).__init__()
        self.embed_dim = embed_dim

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False

        self.encoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)

        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device)
        z = torch.zeros_like(x[:,0:1,:])
        att   = self.encoder_attention
        mover = self.encoder_mover.weight
        mover = (z+1)*mover[None,:,:]
        norm = self.norm1

        g    = torch.cat([z, mover,],dim=1)
        for i in range(x.shape[1]):
            z    = z + x[:,i:i+1,:]
            g    = torch.cat([z, mover,],dim=1)
            dg   = att(g,g,g)
            g    = norm(g+dg)
            z    = g[:,:1]
            mover= g[:,1:]
        return g

    def forward(self, hidden_state, output_sequence):
        x     = self.embedding_out(output_sequence).to(self.device)
        # z     = hidden_state
        xs    = x.shape
        z     = torch.zeros([xs[0],1,xs[2]])
        norm  = self.norm2
        att   = self.decoder_attention
        att2  = self.decoder_attention
        mover = self.decoder_mover.weight
        # mover = self.decoder_mover.weight
        # mover = (torch.ones_like(z[:,0:1])+1)*mover[None,:,:]
        mover = (torch.ones_like(z[:,0:1]))*mover[None,:,:]

        mover = torch.cat([ hidden_state[:,1:], mover,],dim=1)
        z     = hidden_state[:,:1]
        g     = torch.cat([ z, mover,],dim=1)
        for i in range(x.shape[1]):
            z    = z + x[:,i:i+1,:]
            g    = torch.cat([z, mover,],dim=1)
            dg   = att(g,g,g)

            dg2  = att2(g,g,g)
            g    = norm(g+dg+dg2)
            z    = g[:,:1]
            mover= g[:,1:]

        y = self.vocab(z)
        return g[:,:hidden_state.shape[1]],y

    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)

class JointMixtureRNN(nn.Module):
    '''
    Apply an enriched dynamics by splitting the phase space into different regions
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(JointMixtureRNN, self).__init__()
        self.embed_dim = embed_dim

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False

        self.encoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.encoder_mover_2 = nn.Linear(embed_dim,mixture_count,)
        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_mover_2 = nn.Linear(embed_dim,mixture_count,)

        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.encoder_attention_2 = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention_2 = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device)
        z = torch.zeros_like(x[:,0:1,:])
        xs = x.shape
        z0 = torch.zeros([xs[0],1,xs[2]])
        mover  = self.encoder_mover.weight
        mover  = (z[:,:1,:]+1)*mover[None,:,:]
        mover2 = self.encoder_mover_2.weight
        mover2 = (z[:,:1,:]+1)*mover2[None,:,:]
        for i in range(x.shape[1]):

            ### dynamics from the state itself
            g  = torch.cat([ z[:,0:1,:], mover,],dim=1)
            dg = self.encoder_attention(g[:,0:1],g,g)

            ## dynamics from the presented vector
            g2  = torch.cat([x[:,i:i+1,:], mover2,],dim=1)
            dg2 = self.encoder_attention_2(g2[:,0:1],g2,g2)

            z  = z + dg + dg2
            z  = self.norm1(z)
        return z

    def forward(self, hidden_state, output_sequence):
        '''[TBC]change to be like encoder'''
        x = self.embedding_out(output_sequence).to(self.device)
        z = hidden_state
        norm = self.norm2
        att = self.decoder_attention
        mover = self.decoder_mover.weight
        mover = (torch.zeros_like(z)+1)*mover[None,:,:]

        for i in range(x.shape[1]):
            z = z + x[:,i:i+1,:]
            g  = torch.cat([z, mover,],dim=1)
            dg = att(g,g,g)
            z = z + dg[:,0:1]
            z = norm(z)

        y = self.vocab(z)
        return z,y

    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)

class SecondOrderMixtureRNN(nn.Module):
    '''
    Apply an enriched dynamics by splitting the phase space into different regions
    '''
    def __init__(self, embed_dim, mixture_count, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(SecondOrderMixtureRNN, self).__init__()
        self.embed_dim = embed_dim

        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)

        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False

        self.encoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.encoder_mover_2 = nn.Linear(embed_dim,mixture_count,)
        self.decoder_mover = nn.Linear(embed_dim,mixture_count,)
        self.decoder_mover_2 = nn.Linear(embed_dim,mixture_count,)

        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.encoder_attention_2 = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention_2 = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device)
        # z = torch.zeros_like(x[:,0:1,:])
        xs = x.shape
        z0 = torch.zeros([xs[0],1,xs[2]])
        z1 = torch.zeros([xs[0],1,xs[2]])

        norm   = self.norm1

        att   = self.encoder_attention
        att2  = self.encoder_attention_2

        mover  = self.encoder_mover.weight
        mover  = (z0[:,:1,:]+1)*mover[None,:,:]

        mover2 = self.encoder_mover_2.weight
        mover2 = (z0[:,:1,:]+1)*mover2[None,:,:]
        for i in range(x.shape[1]):
            z0 = z0 + x[:,i:i+1,:]
            ### dynamics from the current state itself
            g   = torch.cat([ z0, mover,],dim=1)
            dg  = att(g[:,0:1],g,g)

            ## dynamics from the past state vector
            g2  = torch.cat([ z1, mover2,],dim=1)
            dg2 = att2(g2[:,0:1],g2,g2)

            z0  = z0 + dg + dg2
            z0  = norm(z0)
            z1  = z0
        return torch.cat((z0,z1),dim=1)

    def forward(self, hidden_state, output_sequence):
        x = self.embedding_out(output_sequence).to(self.device)
        xs = x.shape
        if hidden_state.shape[1]!=2:
            # hidden_state =
            hidden_state = torch.cat([hidden_state]*2,dim=1)
        z0 = hidden_state[:,0:1,:]
        z1 = hidden_state[:,1:2,:]


        norm = self.norm2

        att  = self.decoder_attention
        att2 = self.decoder_attention_2

        mover = self.decoder_mover.weight
        mover = (z0[:,:1]*0+1)*mover[None,:,:]

        mover2 = self.decoder_mover_2.weight
        mover2 = (z0[:,:1]*0+1)*mover2[None,:,:]


        for i in range(x.shape[1]):
            z0  = z0 + x[:,i:i+1,:]
            ### dynamics from the current state itself
            g   = torch.cat([ z0, mover,],dim=1)
            dg  = att(g[:,0:1],g,g)

            ## dynamics from the past state vector
            g2  = torch.cat([ z1, mover2,],dim=1)
            dg2 = att2(g2[:,0:1],g2,g2)

            z0  = z0 + dg + dg2
            z0  = norm(z0)
            z1  = z0
        y = self.vocab(z0)
        return torch.cat((z0,z1),dim=1),y

    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)


class ExtendedMixtureRNN(nn.Module):
    def __init__(self, embed_dim, vocab_size,output_vocab_size,max_context_length,CUDA=False):
        super(ExtendedMixtureRNN, self).__init__()
        self.hidden_dim = embed_dim
        self.embed_dim = embed_dim
        self.embedding     = Embeddings(vocab_size,embed_dim,CUDA=CUDA)
        self.embedding_out = Embeddings(output_vocab_size,embed_dim, CUDA=CUDA)
        self.vocab = VocabLogits(embed_dim,output_vocab_size,CUDA=CUDA)
        self.transfer = nn.Linear(embed_dim,embed_dim)
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
        self.encoded = False
        self.encoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_attention = SelfAttention(embed_dim,embed_dim,embed_dim,is_value_embed = 1,mask=False,CUDA=CUDA)
        self.decoder_transfer = nn.Linear(embed_dim*2,embed_dim,)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def encode(self,input_sequence):
        x = self.embedding(input_sequence).to(self.device)
        # z = torch.zeros_like(x[:,0:5,:])
        c = torch.zeros_like(x[:,0:1,:])

        for i in range(x.shape[1]):
            x[:,:,0] = 1   ## 0 is 1 for input
            z[:,:,0] = 0

            z[:,:,1] = 0
            x[:,:,1] = 0
            c[:,:,:] = 0   ## 1 is 1 for context
            c[:,:,1] = 1

            g = torch.cat([z,c,x],dim=1)
            dg =  self.encoder_attention(g,g,g)
            g = g + dg
            g = self.norm1(g)

        return z
        # return z
        # pass
    # def
    def forward(self, hidden_state, output_sequence):
        x = self.embedding_out(output_sequence).to(self.device)

        k = hidden_state
        z = torch.zeros([x.shape[0],1,x.shape[2]])
        for i in range(x.shape[1]):
            z = z + x[:,i:i+1,:]
            z = self.decoder_transfer(torch.cat([z,k],dim=2)) + z
            # [:,:,self.embed_dim]
            z = self.norm2(z)
        y = self.vocab(z)
        return z,y

    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)


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
        self.encode_out = z = self.encoder(x)
        self.encoded = True
        return z

    def forward(self, hidden,output_sequence):
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
            return (hidden, x)

            # embedding = self.embedding2(output_sequence)
            # return self.decoder(self.encode_out,embedding)
    def step(self,hidden_state,x):
        return self.forward(hidden_state,x)
