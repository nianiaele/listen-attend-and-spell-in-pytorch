import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pack_sequence
import numpy as np
import pdb
import configuration
from configuration import dataBasePath,device,epoch_num,batch_size
from DataLoader import WSJDataset,collateFrames
from Model import LasModel
from CrossEntropyLossWithMask import CrossEntropyLossWithMask

def train_epoch():
    total_loss=0
    total_perplexity=0
    batch_num=0
    for batch_id,(x,y,xbounds,ybounds,xLens,yLens,inputy,targety) in enumerate(train_loader):

        batch_num = batch_id + 1

        #give up the last batch
        if len(x)!=batch_size:
            print("batch: ", batch_num)
            continue

        optimizer.zero_grad()

        packed_x=pack_sequence(x)
        # packed_inputy=pack_sequence(inputy)
        # packed_targety=pack_sequence(targety)

        output=model(packed_x,xLens,inputy,targety,yLens)

        loss=criterion(output,targety,yLens)

        loss.backward()

        optimizer.step()

        total_loss+=loss

        total_perplexity+=torch.exp(loss)

        print("batch: ",batch_num)

        del x,y,xbounds,ybounds

    return total_loss/batch_num, total_perplexity/batch_num




model=LasModel()
train_data_set=WSJDataset(dataBasePath+'train.npy',dataBasePath+'newTrainY.npy')
train_loader=DataLoader(train_data_set,shuffle=False,batch_size=configuration.batch_size,collate_fn=collateFrames,num_workers=2)
optimizer=torch.optim.Adam(model.parameters(),lr=configuration.learning_rate,weight_decay=1e-6)

criterion=CrossEntropyLossWithMask()

for epoch in range(epoch_num):
    avg_loss,avg_perplexity=train_epoch()
    print("epcoh "+str(epoch)+" average_loss is "+str(avg_loss.item())+" average_perplexity is "+str(avg_perplexity.item()))
