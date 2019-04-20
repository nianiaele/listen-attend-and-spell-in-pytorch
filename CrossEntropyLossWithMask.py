import torch.nn as nn
import torch
from torch.nn.functional import pad
from configuration import batch_size,device
from torch.autograd import Variable

class CrossEntropyLossWithMask(nn.CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLossWithMask,self).__init__(reduction='none')

    def forward(self, input, target,targe_length):

        #pad the shorter one if target and input are of different length
        if target.size(1)!=input.size(1):
            if target.size(1)>input.size(1):
                distance=target.size(1)-input.size(1)
                pd=(0,0,0,distance)
                input=pad(input, pd, "constant", 0)

                distance=5
                input[:,input.size(1)-distance:,32]=torch.ones(input[:,input.size(1)-distance:,32].size())*32
            elif target.size(1)<input.size(1):
                distance=input.size(1)-target.size(1)
                pd=(0,distance)
                target=pad(target,pd,"constant",32)
        loss=super(CrossEntropyLossWithMask,self).forward(input.view(-1,input.size()[2]), target.contiguous().view(-1).type(torch.LongTensor))


        max_length=input.size(1)

        loss_mask=torch.FloatTensor(batch_size,max_length).zero_()

        for i in range(batch_size):
            this_length=int(targe_length[i])
            loss_mask[i,0:this_length]=torch.ones(1,this_length).float()

        loss_mask=Variable(loss_mask).to(device)

        loss=loss.view(batch_size,max_length)

        loss=loss*loss_mask

        # loss=torch.sum(loss,dim=1)

        loss=torch.mean(loss)

        return loss




