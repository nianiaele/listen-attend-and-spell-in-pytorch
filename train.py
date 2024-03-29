import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pack_sequence
import numpy as np
import pdb
import configuration
from configuration import dataBasePath,device,epoch_num,batch_size,clip_value
from DataLoader import WSJDataset,collateFrames
from Model import LasModel
from CrossEntropyLossWithMask import CrossEntropyLossWithMask
from util import plot_grad_flow,show_attention_weights
from Dictionary import to_string
from torch.nn.utils import clip_grad_value_

def train_epoch(epoch_num):
    total_loss=0
    total_perplexity=0
    batch_num=0
    model.is_train=True
    model.to(device)
    torch.autograd.set_detect_anomaly(True)
    for batch_id,(x,y,xbounds,ybounds,xLens,yLens,inputy,targety) in enumerate(train_loader):

        batch_num = batch_id + 1

        # print(len(x))
        #give up the last batch
        if len(x)!=batch_size:
            print("give up batch: ", batch_num)
            continue

        optimizer.zero_grad()

        # x.to(device)
        packed_x=pack_sequence(x)

        packed_x=packed_x.to(device)
        xLens=xLens.to(device)
        yLens=yLens.to(device)
        inputy=inputy.to(device)

        # print(packed_x.dtype)


        output,stack_attention=model(packed_x,xLens,inputy,targety,yLens)


        loss=criterion(output,targety,yLens)

        loss.backward()

        clip_grad_value_(model.parameters(), clip_value)

        plot_grad_flow(model.named_parameters())


        optimizer.step()

        total_loss+=loss

        total_perplexity+=torch.exp(loss)


        if batch_id%configuration.print_cut==0:
            print("batch: ", batch_num)
            print("output is:",to_string(output))
            show_attention_weights(stack_attention)

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

            packed_x=packed_x.to(device)
            xLens=xLens.to(device)
            inputy=inputy.to(device)
            targety=targety.to(device)
            yLens=yLens.to(device)

            output,stack_attention = model(packed_x, xLens, inputy, targety, yLens)
            loss = criterion(output, targety, yLens)
            total_dev_loss += loss
            total_dev_perplexity += torch.exp(loss)


    model.is_train=True
    model.to('cpu')
    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_label':optimizer.state_dict()
    }, "./myModel7"+str(epoch_num))
    print("model saved")

    return total_loss/batch_num, total_perplexity/batch_num, total_dev_loss/dev_batch_number, total_dev_perplexity/dev_batch_number


model=LasModel()
train_data_set=WSJDataset(dataBasePath+'train.npy',dataBasePath+'newTrainY.npy')
train_loader=DataLoader(train_data_set,shuffle=True,batch_size=configuration.batch_size,collate_fn=collateFrames,num_workers=32)

dev_dataset=WSJDataset(dataBasePath+'dev.npy',dataBasePath+'newDevY.npy')
dev_loader=DataLoader(dev_dataset,shuffle=False,batch_size=configuration.batch_size,collate_fn=collateFrames,num_workers=32)

optimizer=torch.optim.Adam(model.parameters(),lr=configuration.learning_rate,weight_decay=1e-6)
# optimizer=torch.optim.SGD(model.parameters(),lr=configuration.learning_rate,weight_decay=1e-6)

criterion=CrossEntropyLossWithMask().to(device)

if configuration.is_pretrain==True:
    print("loading model")
    check_point = torch.load("./myModel64", map_location='cpu')
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_label'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # model=model.to(device)

for epoch in range(epoch_num):
    train_loss,perplexity,dev_loss,dev_perplexity=train_epoch(epoch)
    print("epcoh "+str(epoch)+" average_loss: "+str(train_loss.item())+" average_perplexity: "+str(perplexity.item())
          +" avg_dev_loss: "+str(dev_loss.item())+" avg_dev_perplexity: "+str(dev_perplexity.item()))
    if configuration.teacher_forcing<-0.1:
        configuration.teacher_forcing = configuration.teacher_forcing + 0.01
    print("teacher forcing set to ",configuration.teacher_forcing)
