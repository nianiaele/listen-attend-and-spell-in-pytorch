import torch
import torch.nn as nn
import configuration
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.embedding=nn.Embedding(num_embeddings=configuration.dictionary_length,embedding_dim=configuration.embedding_size)

        self.cell1=nn.LSTMCell(input_size=configuration.kqv_size,hidden_size=configuration.speller_hidden_size)
        self.cell2=nn.LSTMCell(input_size=configuration.speller_hidden_size,hidden_size=configuration.speller_hidden_size)
        self.cell3=nn.LSTMCell(input_size=configuration.speller_hidden_size,hidden_size=configuration.kqv_size)

    #inputs(length, batch_size, dim)
    def forward(self,inputs,keys,values):


    def cellForward(self,input):




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
        metrix.view(configuration.batch_size,new_length,metrix.size(2)*2)

        metrix=metrix.transpose(0,1)

        half_length=[l//2 for l in length]

        returnPack=pack_padded_sequence(metrix,half_length)

        return returnPack



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.bilstm=nn.LSTM(input_size=configuration.frame_dim,hidden_size=configuration.listener_hidden_size,
                            batch_first=False,bidirectional=True)
        self.pbilstm1=nn.LSTM(input_size=configuration.listener_hidden_size*2,hidden_size=configuration.listener_hidden_size,
                              batch_first=False,bidirectional=True)
        self.pbilstm2=nn.LSTM(input_size=configuration.listener_hidden_size*2,hidden_size=configuration.listener_hidden_size,
                              batch_first=False,bidirectional=True)
        self.pbilstm3=nn.LSTM(input_size=configuration.listener_hidden_size*2,hidden_size=configuration.listener_hidden_size,
                              batch_first=False,bidirectional=True)

        self.pooling=Pooling()

        self.key_linear=nn.Linear(in_features=configuration.listener_hidden_size*2,
                                  out_features=configuration.kqv_size)
        self.value_linear = nn.Linear(in_features=configuration.listener_hidden_size * 2,
                                    out_features=configuration.kqv_size)

    #inputs is a packed sequence, on device
    def forward(self,inputs,length):
        output1,_=self.bilstm(inputs)

        output2=self.pooling(output1)

        output3=self.pbilstm1(output2)

        output4=self.pooling(output3)

        output5=self.pbilstm2(output4)

        output6=self.pooling(output5)

        output7=self.pbilstm3(output6)

        keys=self.key_linear(output7)

        values=self.value_linear(output7)

        return keys,values




