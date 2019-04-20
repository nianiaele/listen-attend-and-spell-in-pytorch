import torch
import torch.nn as nn
import configuration
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Variable
from configuration import batch_size,kqv_size,device,teacher_forcing
import torch.nn.functional as F
import random

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.embedding=nn.Embedding(num_embeddings=configuration.dictionary_length,embedding_dim=configuration.embedding_size)

        self.cell1=nn.LSTMCell(input_size=configuration.kqv_size*2,hidden_size=configuration.speller_hidden_size)
        self.cell2=nn.LSTMCell(input_size=configuration.speller_hidden_size,hidden_size=configuration.speller_hidden_size)
        self.cell3=nn.LSTMCell(input_size=configuration.speller_hidden_size,hidden_size=configuration.kqv_size)

        hidden_dim=configuration.speller_hidden_size
        attention_dim=configuration.kqv_size
        batch_size=configuration.batch_size

        self.h_0_1 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c_0_1 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.h_0_2 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c_0_2 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.h_0_3 = nn.Parameter(torch.zeros(1, attention_dim))
        self.c_0_3 = nn.Parameter(torch.zeros(1, attention_dim))

        self.h1 = self.h_0_1.expand(batch_size, -1)
        self.h2 = self.h_0_2.expand(batch_size, -1)
        self.h3 = self.h_0_3.expand(batch_size, -1)

        self.c1 = self.c_0_1.expand(batch_size, -1)
        self.c2 = self.c_0_2.expand(batch_size, -1)
        self.c3 = self.c_0_3.expand(batch_size, -1)

    #inputs(length, batch_size, dim)
    def forward(self,inputs,context,char_index):
        inputs=inputs.type(torch.LongTensor)
        inputs=self.embedding(inputs)

        context=context.squeeze()

        h=torch.cat((inputs,context),1)

        if char_index == 0:
            self.h1 = self.h_0_1.expand(configuration.batch_size, -1)
            self.h2 = self.h_0_2.expand(configuration.batch_size, -1)
            self.h3 = self.h_0_3.expand(configuration.batch_size, -1)

            self.c1 = self.c_0_1.expand(configuration.batch_size, -1)
            self.c2 = self.c_0_2.expand(configuration.batch_size, -1)
            self.c3 = self.c_0_3.expand(configuration.batch_size, -1)

        self.h1, self.c1 = self.cell1(h, (self.h1, self.c1))

        self.h2, self.c2 = self.cell2(self.h1, (self.h2, self.c2))

        self.h3, self.c3 = self.cell3(self.h2, (self.h3, self.c3))

        return self.h3





class Pooling(nn.Module):
    def __init__(self):
        super(Pooling,self).__init__()

    def forward(self, input):
        metrix,length=pad_packed_sequence(input)
        max_length=metrix.size(0)
        new_length=max_length//2

        metrix=metrix.transpose(0,1)

        if max_length%2==1:
            metrix=metrix[:,0:max_length-1,:]

        # print(metrix.size())
        metrix=metrix.contiguous().view((-1,new_length,metrix.size(2)*2))

        metrix=metrix.transpose(0,1)

        half_length=[l//2 for l in length]

        returnPack=pack_padded_sequence(metrix,half_length)

        return returnPack



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.bilstm=nn.LSTM(input_size=configuration.frame_dim,hidden_size=configuration.listener_hidden_size,
                            batch_first=False,bidirectional=True)
        self.pbilstm1=nn.LSTM(input_size=configuration.listener_hidden_size*4,hidden_size=configuration.listener_hidden_size,
                              batch_first=False,bidirectional=True)
        self.pbilstm2=nn.LSTM(input_size=configuration.listener_hidden_size*4,hidden_size=configuration.listener_hidden_size,
                              batch_first=False,bidirectional=True)
        self.pbilstm3=nn.LSTM(input_size=configuration.listener_hidden_size*4,hidden_size=configuration.listener_hidden_size,
                              batch_first=False,bidirectional=True)

        self.test=nn.LSTM(input_size=configuration.frame_dim,hidden_size=configuration.listener_hidden_size,
                            batch_first=False,bidirectional=False)

        self.pooling=Pooling()

        self.key_linear=nn.Linear(in_features=configuration.listener_hidden_size*2,
                                  out_features=configuration.kqv_size)
        self.value_linear = nn.Linear(in_features=configuration.listener_hidden_size * 2,
                                    out_features=configuration.kqv_size)

    #inputs is a packed sequence, on device
    def forward(self,inputs,length):
        output1,_=self.bilstm(inputs)

        # output1, _ = self.test(inputs)

        output2=self.pooling(output1)

        output3,_=self.pbilstm1(output2)

        output4=self.pooling(output3)

        output5,_=self.pbilstm2(output4)

        output6=self.pooling(output5)

        output7,_=self.pbilstm3(output6)

        paded_output7,_=pad_packed_sequence(output7)

        keys=self.key_linear(paded_output7)

        values=self.value_linear(paded_output7)

        return keys,values

class LasModel(nn.Module):
    def __init__(self,is_train=True):
        super(LasModel,self).__init__()

        self.is_train=is_train

        self.encoder=Encoder()
        self.decoder=Decoder()

        self.linear1=nn.Linear(in_features=configuration.speller_hidden_size,out_features=configuration.kqv_size)

        self.relu=nn.ReLU()

        self.linear2=nn.Linear(in_features=configuration.kqv_size,out_features=configuration.output_dim)

        self.log_softmax = nn.Softmax()

    def forward(self,x_input,x_length,y_input,y_target,y_length):


        keys,values=self.encoder(x_input,x_length)

        context=Variable(torch.FloatTensor(batch_size,1,kqv_size).zero_()).to(device)


        max_x_length_after_pbilstm=keys.size(0)

        max_y_length=torch.max(y_length)

        #keys,values,mask shape[batch_size,max_seq,kqv_dim]
        keys=keys.transpose(0,1)
        values=values.transpose(0,1)

        mask=torch.FloatTensor(batch_size,1,max_x_length_after_pbilstm).zero_()
        for i in range(x_length.size(0)):
            this_length=int(x_length[i])//8
            mask[i,0,0:this_length]=torch.ones(this_length).float()
        mask=Variable(mask).to(device)


        out=[]


        last_logit=None
        for i in range(max_y_length):
            # char=y_input[i]#should be sqeuence?





            if self.is_train==True:
                r=random.uniform(0,1)
                if i == 0:
                    char = torch.LongTensor([32] * batch_size)
                elif r>teacher_forcing:
                    char=y_input[i]
                else:
                    predict = self.log_softmax(last_logit)
                    char = torch.max(predict, dim=1).indices
            else:
                if i == 0:
                    char = torch.LongTensor([32] * batch_size)
                else:
                    predict = self.log_softmax(last_logit)
                    char = torch.max(predict, dim=1).indices


            query=self.decoder(char,context,i)

            energy=torch.bmm(query.unsqueeze(1),keys.transpose(1,2))

            attention=F.softmax(energy,dim=2)

            attention=attention*mask

            attention = attention/torch.sum(attention, 2).unsqueeze(2)

            context = torch.bmm(attention, values)

            context = context.squeeze(1)

            mlp_input = torch.cat((context, query), 1)

            logit = self.linear1( mlp_input )
            logit = self.relu(logit)
            logit = self.linear2(logit)

            #add a softmax?
            out += [logit]
            last_logit=logit

        out=torch.stack(out,1)
        return out