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
    model.is_train=True
    for batch_id,(x,y,xbounds,ybounds,xLens,yLens,inputy,targety) in enumerate(train_loader):

        batch_num = batch_id + 1

        #give up the last batch
        if len(x)!=batch_size:
            print("give up batch: ", batch_num)
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


    #dev loss
    total_dev_loss=0
    total_dev_perplexity=0
    model.is_train=False
    dev_batch_number=0
    for batch_id, (x, y, xbounds, ybounds, xLens, yLens, inputy, targety) in enumerate(dev_loader):

        dev_batch_number = batch_id + 1

        if len(x)!=batch_size:
            continue

        with torch.no_grad():
            packed_x = pack_sequence(x)
            output = model(packed_x, xLens, inputy, targety, yLens)
            loss = criterion(output, targety, yLens)
            total_dev_loss += loss
            total_dev_perplexity += torch.exp(loss)

    model.is_train=True
    model.to('cpu')
    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_label':optimizer.state_dict()
    }, "./myModel")
    print("model saved")

    return total_loss/batch_num, total_perplexity/batch_num, total_dev_loss/dev_batch_number, total_dev_perplexity/dev_batch_number


model=LasModel()
train_data_set=WSJDataset(dataBasePath+'train.npy',dataBasePath+'newTrainY.npy')
train_loader=DataLoader(train_data_set,shuffle=False,batch_size=configuration.batch_size,collate_fn=collateFrames,num_workers=2)

dev_dataset=WSJDataset(dataBasePath+'dev.npy',dataBasePath+'newDevY.npy')
dev_loader=DataLoader(dev_dataset,shuffle=False,batch_size=configuration.batch_size,collate_fn=collateFrames,num_workers=2)

optimizer=torch.optim.Adam(model.parameters(),lr=configuration.learning_rate,weight_decay=1e-6)

criterion=CrossEntropyLossWithMask()

for epoch in range(epoch_num):
    loss,perplexity,dev_loss,dev_perplexity=train_epoch()
    print("epcoh "+str(epoch)+" average_loss: "+str(loss.item())+" average_perplexity: "+str(perplexity.item())
          +" avg_dev_loss: "+str(dev_loss.item())+" avg_dev_perplexity: "+str(dev_perplexity.item()))
