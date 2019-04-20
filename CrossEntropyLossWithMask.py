import torch.nn as nn
import torch
from configuration import batch_size,device
from torch.autograd import Variable

class CrossEntropyLossWithMask(nn.CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLossWithMask,self).__init__(reduction='none')

    def forward(self, input, target,targe_length):
        loss=super(CrossEntropyLossWithMask,self).forward(input.view(-1,input.size()[2]), target.transpose(0,1).contiguous().view(-1).type(torch.LongTensor))


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




