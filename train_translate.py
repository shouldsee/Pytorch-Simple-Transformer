from Dataset.translation_dataset import EnglishToGermanDataset
# from Transformer.transfomer import TransformerTranslator
from Transformer.transfomer import TransformerTranslatorFeng
# ,TransformerTranslatorFeng
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os


"""
Hyperparameters
"""
CUDA = True
CUDA = 1
PRINT_INTERVAL = 5000
VALIDATE_AMOUNT = 10

batch_size = 81
embed_dim = 64
num_blocks = 12
num_heads = 1 #Must be factor of token size
max_context_length = 1000

SAVE_INTERVAL = 1000//batch_size
num_epochs = 1000
learning_rate = 1e-3

device = torch.device('cuda:0' if CUDA else 'cpu')


"""
Dataset
"""
dataset = EnglishToGermanDataset(CUDA=CUDA)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# assert 0
"""
Model
"""
# vocab_size = dataset.german_vocab_len
# vocab_size = dataset.english_vocab_len
# print((dataset.english_vocab_len, dataset.german_vocab_len))
# assert 0
torch.set_default_tensor_type(torch.cuda.FloatTensor if CUDA else torch.FloatTensor)
from Transformer.transfomer import BasicRNN
from Transformer.transfomer import MixtureRNN
from Transformer.transfomer import DynamicMixtureRNN
from Transformer.transfomer import JointMixtureRNN
from Transformer.transfomer import SecondOrderMixtureRNN
from Transformer.transfomer import AnchorMixtureRNN
from Transformer.transfomer import MCMCAnchorMixtureRNN
from Transformer.transfomer import BeamAnchorMixtureRNN
from Transformer.transfomer import AnchorOnlyMixtureRNN
# from Transformer.transfomer import MCMCAnchorOnlyMixtureRNN
from Transformer.transfomer import BeamMixtureRNN
from Transformer.transfomer import decode_with_target
# model = TransformerTranslatorFeng(embed_dim,num_blocks,num_heads, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model =  BasicRNN(embed_dim, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)

# model = SecondOrderMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model = DynamicMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model = MixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model = AnchorOnlyMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model = AnchorMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model = BeamAnchorMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
# model = MCMCAnchorMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)
model = BeamMixtureRNN(embed_dim,num_blocks, dataset.english_vocab_len, dataset.german_vocab_len,max_context_length=max_context_length,CUDA=CUDA).to(device)

"""
Loss Function + Optimizer
"""
def sqrt_sum( output_log , target):
    # print(output_log.shape)
    loss = torch.exp(output_log/2.) * target
    loss = torch.sum(loss,axis=-1)
    loss = -torch.mean(loss)
    #(output-target*2)**3)
    return loss



def loss_kldiv(pred, target):

    ## pred
    # print(input[0][0][:5],target[0][0][:5])
    # loss = torch.exp(pred) * (pred- target)
    loss = - (pred) * target
    # ### encourage
    loss = loss.sum(dim=(1,2)).mean(dim=0)


    # loss = - torch.exp(pred) * target
    # # ### encourage
    # loss = loss.sum(dim=(1,2)).mean(dim=0)
    # # loss = - pred * target
    return loss
    # loss.sum(dim=(1,2)).mean(dim=0)

baseline = [0.]
def loss_with_baseline(out, target):
    pred,out_prob_bychain = out

    ## pred
    # print(input[0][0][:5],target[0][0][:5])
    # loss = torch.exp(pred) * (pred- target)
    # print(pred[0,:3,:10])
    loss = -(pred) * (target - 0.000)
    # ### encourage
    loss = (loss.sum(dim=(1,2)) - baseline[0]).mean(dim=0)
    # loss = - torch.exp(pred) * target
    ##### encourage
    # loss = loss.sum(dim=(1,2)).mean(dim=0)
    # # loss = - pred * target
    return loss

def loss_by_chain(out, target):
    pred,out_prob_bychain = out
    ###### pred
    loss = - (out_prob_bychain) * (target[:,:,None,:] - 0.000)
    loss = loss.sum(dim=(1,3)).mean(dim=(0,1))
    ###### encourage
    return loss

# criterion = sqrt_sum
criterion = lambda out,target,f=nn.KLDivLoss(reduction='batchmean'): f(out[0],target)
# criterion = loss_by_chain
optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adagrad( model.parameters(), lr=learning_rate)
"""
Load From Checkpoint
"""
import sys

import glob
from pprint import pprint
os.makedirs('Checkpoints') if not os.path.exists('Checkpoints') else None
criterion
if '--LOAD' in sys.argv:
    LOAD = sys.argv[sys.argv.index('--LOAD')+1]
else:
    LOAD = None


if LOAD == 'auto':
    res = glob.glob('Checkpoints/*pkl')
    getLoad = lambda x:int(x.replace('Checkpoints/Checkpoint','').split('.')[0])
    res = sorted(res,key=getLoad)[-1]
    LOAD= getLoad(res)
elif LOAD is not None:
    LOAD = int(LOAD)
else:
    LOAD = -1


IS_GREEDY = '--greedy' in sys.argv
STRICT_LOAD = '--nostrict' not in sys.argv
def main():
    if(LOAD!=-1):
        checkpoint   =  torch.load(os.path.join("Checkpoints","Checkpoint"+str(LOAD)+".pkl"))
        test_losses  =  checkpoint["test_losses"]
        train_losses =  checkpoint["train_losses"]
        num_steps    =  checkpoint["num_steps"]
        x = checkpoint["model"]
        xx = {}
        for k,v in x.items():
            if k in dict(model.named_parameters()):
                xx[k] = v
            else:
                pass
                # print(k)
        x = xx
        # x = {k:v for k,v in x.items() if k in dict(model.named_parameters())}
        model.load_state_dict(x,strict=STRICT_LOAD)
        # optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        test_losses = []
        train_losses = []
        num_steps = 0
    """
    Train Loop
    """
    for epoch in range(num_epochs):
        running_loss = []
        running_test_loss = []
        dataset.train()
        """
        TRAIN LOOP
        """

        # print(model.state_dict().keys())
        print(model.__class__.__name__)
        for idx,item in enumerate(tqdm(dataloader)):
            """
            ============================================================
            """
            model.train()
            # print(model.encoder_mover.weight.T[:,:5])
            # .shape)
            ###################
            #Zero Gradients
            model.zero_grad()
            optimizer.zero_grad()
            ###################

            ###################
            #Encode English Sentence
            # perm = np.random.permutation(item["english"][:,1:-1].shape[1])
            hidden = model.encode(item["english"][:,:][:,:])
            # print(item["german"][:,1:-1][:1])
            ###################

            item['german'] = item['german'][:,1:]
            item['logits'] = item['logits'][:,1:]
            item['logit_mask'] = item['logit_mask'][:,1:]
            out = decode_with_target(model,  hidden, item['german'],IS_GREEDY)
            all_outs,all_outs_by_chain = out

            ###################
            #BackProp
            loss = criterion(out, (item["logits"]* item["logit_mask"]))
            loss.backward()
            optimizer.step()

            ###################
            running_loss.append(loss.item())
            num_steps +=1
            """
            ============================================================
            """
            if(num_steps % PRINT_INTERVAL ==0 or idx==len(dataloader)-1):
                """
                Validation LOOP
                """

                all_outs.detach().cpu()
                item["logits"].detach().cpu()
                dataset.test()
                model.eval()
                with torch.no_grad():
                    if '--debug' in sys.argv:
                        sents =[]
                        sents += ['Herr Präsident und Herr Kolleginnen , Herr Kollegen' ] #+ ' <start>'*30
                        sents += ['Herr Herr Herr Herr'  + ' <start>'*30]
                        sents += ['Herr Präsident'  + ' <start>'*30]
                        # sents += ['Herr Präsident , Herr Kollegen' ] #+ ' <start>'*30
                        sents = [
                        torch.tensor([dataset.english_vocab_reversed.index(xx) for xx in sent.replace(' ','| |').split('|')],dtype=torch.long).cpu()
                        for sent in sents]
                        x = dataset.english_sentences_test[0:3] + sents
                        x = [xx[:14] for xx in x]
                        # x = torch.cat(x,dim=0).to(device)
                        x = torch.stack(x,dim=0)[:,:].to(device)
                        print([dataset.english_vocab_reversed[xx] for xx in x.cpu().detach().numpy().ravel()])

                        hidden = model.encode(x)
                        out = decode_with_target(model, hidden, x, is_greedy=1)
                        all_outs,all_outs_by_chain = out

                        for vidx in range(len(x)):
                            print('==============')
                            print('INPUT:'+ ','.join([dataset.english_vocab_reversed[xx].__repr__() for xx in x[vidx].cpu().detach().numpy().ravel()]))
                            print('PRED:'+ ','.join([dataset.german_vocab_reversed[xx].__repr__() for xx in all_outs[vidx].argmax(-1).cpu().detach().numpy().ravel()]))
                            pred_tok = [dataset.german_vocab_reversed[xx].__repr__() for xx in all_outs[vidx].argmax(-1).cpu().detach().numpy().ravel()]
                            # print("PRED: ",(dataset.logit_to_sentence(all_outs[vidx])))
                            if isinstance(hidden,tuple) and hasattr(model,'pointers'):
                                # list(map( lambda x:[print(('%d, '%xx).rjust(5,' '),end='') for xx in x] + [print()],(hidden[1][vidx,:,:5].T.cpu().numpy()*10).astype(int).tolist()))
                                # ( z,anchor_value, anchor_att_ret, bpointer,zs )  = hidden
                                for toki in range(pred_tok.__len__()):
                                    bpointer = (model.pointers//model.mixture_count)[vidx,toki].cpu().detach().numpy()
                                    fpointer = (model.pointers%model.mixture_count)[vidx,toki].cpu().detach().numpy()
                                    print(fpointer,pred_tok[toki],bpointer)
                                    # print((,pred_tok[toki])
                                    # print(model.pointers.shape,toki,all_outs.shape,)
                                # list(map( lambda x:[print(('%d, '%xx).rjust(5,' '),end='') for xx in x] + [print()],(hidden[1][vidx,:,:5].T.cpu().numpy()*10).astype(int).tolist()))

                    for jdx,item in enumerate(dataloader_test):
                        item['german'] = item['german'][:,1:]
                        item['logits'] = item['logits'][:,1:]
                        item['logit_mask'] = item['logit_mask'][:,1:]

                        hidden = model.encode(item["english"][:,:])
                        g = item["german"].shape
                        x = torch.zeros( [g[0],g[1],],dtype=torch.long ).to(device)

                        out = decode_with_target(model, hidden, item["german"], IS_GREEDY)
                        all_outs, all_outs_by_chain = out

                        item["logits"] = item["logits"] * item["logit_mask"]
                        loss = criterion(out,item["logits"])
                        running_test_loss.append(loss.item())
                        if(jdx==VALIDATE_AMOUNT):
                            break

                avg_test_loss = np.array(running_test_loss).mean()
                test_losses.append(avg_test_loss)
                avg_loss      = np.array(running_loss).mean()
                train_losses.append(avg_loss)
                for vidx in [0,1]:
                    print("===")
                    # print()
                    print("LABEL: ",dataset.logit_to_sentence(item["logits"][vidx]))
                    print("PRED: ",(dataset.logit_to_sentence(all_outs[vidx])))
                    print()
                    # print("===")


                # print((model.decoder_anchor_disp_list[5].weight[:15,:15].cpu().detach()*100).numpy().astype(int))
                # _x = (model.decoder_anchor_disp_list[5].bias*100)
                # print(_x.cpu().detach().numpy().astype(int))
                # [:5,:5])
                # print((model.decoder_anchor_disp_list[6].weight[:5,:5].cpu().detach()*100).numpy().astype(int))
                # print(model.decoder_anchor_disp_list[6].bias)
                #
                # print((model.decoder_mover.weight[4:7,:].cpu().detach()*100).numpy().astype(int))
                # print()
                #
                # f = model.decoder_anchor_disp_list[5]
                # _x = (f(model.decoder_mover.weight[4:7,:])*100).cpu().detach().numpy().astype(int)
                # print(_x)
                # print()
                #
                # _x = (model.decoder_anchor_disp_list[5](model.decoder_mover.weight[4:7,:]-model.decoder_mover.weight[4:7,:])*100).cpu().detach().numpy().astype(int)
                # print(_x)
                # print()
                #
                # ,model.decoder_anchor_disp_list[5].weight,)
                # print("PRED: ",list(dataset.logit_to_sentence(all_outs[0])))
                #

                print(f"TRAIN LOSS {avg_loss} | EPOCH {epoch}")
                print(f"TEST LOSS {avg_test_loss} | EPOCH {epoch}")
                print("BACK TO TRAINING:")
                dataset.train()
                # model.encoder.show_matrix(__file__+'.html')

            if(num_steps % SAVE_INTERVAL ==0):
                torch.save({
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "num_steps":num_steps,
                    "train_losses":train_losses,
                    "test_losses":test_losses
                },os.path.join("Checkpoints","Checkpoint"+str(num_steps)+".pkl"))

if __name__ =='__main__':
    main()
