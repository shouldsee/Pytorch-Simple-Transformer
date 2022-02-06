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
CUDA = False
PRINT_INTERVAL = 5000
VALIDATE_AMOUNT = 10

batch_size = 128
embed_dim = 64
num_blocks = 4
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
model = TransformerTranslatorFeng(embed_dim,num_blocks,num_heads, dataset.english_vocab_len, dataset.german_vocab_len,
max_context_length=max_context_length,
CUDA=CUDA).to(device)

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
criterion = sqrt_sum

optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate)
criterion = nn.KLDivLoss(reduction='batchmean')
"""
Load From Checkpoint
"""
import sys
if '--auto' in sys.argv:
    LOAD_GET_AUTO = 1
else:
    LOAD_GET_AUTO = 0

import glob
from pprint import pprint
os.makedirs('Checkpoints') if not os.path.exists('Checkpoints') else None

if LOAD_GET_AUTO:
    res = glob.glob('Checkpoints/*pkl')
    getLoad = lambda x:int(x.replace('Checkpoints/Checkpoint','').split('.')[0])
    res = sorted(res,key=getLoad)[-1]
    LOAD= getLoad(res)
else:
    LOAD = -1

if '--LOAD' in sys.argv:
    LOAD = sys.argv[sys.argv.index('--LOAD')+1]
    LOAD = int(LOAD)

def main():

    if(LOAD!=-1):
        checkpoint   = torch.load(os.path.join("Checkpoints","Checkpoint"+str(LOAD)+".pkl"))
        test_losses  = checkpoint["test_losses"]
        train_losses = checkpoint["train_losses"]
        num_steps    = checkpoint["num_steps"]
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

        model.load_state_dict(x)
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
        for idx,item in enumerate(tqdm(dataloader)):
            """
            ============================================================
            """
            model.train()

            ###################
            #Zero Gradients
            model.zero_grad()
            optimizer.zero_grad()
            ###################

            ###################
            #Encode English Sentence
            # perm = np.random.permutation(item["english"][:,1:-1].shape[1])
            model.encode(item["english"][:,1:-1][:,:])
            # print(item["german"][:,1:-1][:1])
            ###################

            ###################
            #Output German, One Token At A Time
            all_outs = torch.tensor([],requires_grad=True).to(device)
            # print(item["german"].shape[1])
            # all_outs = model(model.encode_out)[:,:]
            # item["german"].shape[1]-1]
            for i in range(item["german"].shape[1]-1):
                out = model(item["german"][:,:i+1])
                all_outs = torch.cat((all_outs,out),dim=1)
            ###################

            ###################
            #Mask Out Extra Padded Tokens In The End(Optional)
            # outputRotation = torch.eye(all_outs.shape[1])
            # print(outputRotation.shape)
            # [1])
            all_outs = all_outs * item["logit_mask"][:,:-1]
            # item["logits"] = item["logits"] * item["logit_mask"]

            ###################

            ###################
            #BackProp
            loss = criterion(all_outs,item["logits"][:,:-1,:])
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
                    for jdx,item in enumerate(dataloader_test):
                        model.encode(item["english"][:,1:-1])
                        all_outs = torch.tensor([],requires_grad=True).to(device)
                        for i in range(item["german"].shape[1]-1):
                            out = model(item["german"][:,:i+1])
                            all_outs = torch.cat((all_outs,out),dim=1)

                        # all_outs = model(model.encode_out)[:,:]
                        # all_outs = all_outs * item["logit_mask"][:,1:-2,:]

                        item["logits"] = item["logits"] * item["logit_mask"]
                        loss = criterion(all_outs,item["logits"][:,:-1,:])
                        running_test_loss.append(loss.item())
                        if(jdx==VALIDATE_AMOUNT):
                            break
                avg_test_loss = np.array(running_test_loss).mean()
                test_losses.append(avg_test_loss)
                avg_loss = np.array(running_loss).mean()
                train_losses.append(avg_loss)
                print("LABEL: ",dataset.logit_to_sentence(item["logits"][0]))
                print("===")
                print("PRED: ",dataset.logit_to_sentence(all_outs[0]))
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
main()
