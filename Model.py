import torch
import torch.nn as nn
import configuration
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from configuration import batch_size, device, teacher_forcing
import torch.nn.functional as F
import random
import numpy as np


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=configuration.dictionary_length,
                                      embedding_dim=configuration.embedding_size)

        self.cell1 = nn.LSTMCell(input_size=configuration.kqv_size + configuration.embedding_size,
                                 hidden_size=configuration.speller_hidden_size)
        self.cell2 = nn.LSTMCell(input_size=configuration.speller_hidden_size, hidden_size=configuration.kqv_size)
        self.cell3 = nn.LSTMCell(input_size=configuration.kqv_size, hidden_size=configuration.kqv_size)

        hidden_dim = configuration.speller_hidden_size
        attention_dim = configuration.kqv_size
        batch_size = configuration.batch_size
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.h_0_1 = Variable(torch.zeros(1, self.hidden_dim))
        self.c_0_1 = Variable(torch.zeros(1, self.hidden_dim))
        self.h_0_2 = Variable(torch.zeros(1, self.attention_dim))
        self.c_0_2 = Variable(torch.zeros(1, self.attention_dim))
        self.h_0_3 = Variable(torch.zeros(1, self.attention_dim))
        self.c_0_3 = Variable(torch.zeros(1, self.attention_dim))

        self.h1 = self.h_0_1.expand(batch_size, -1)
        self.h2 = self.h_0_2.expand(batch_size, -1)
        self.h3 = self.h_0_3.expand(batch_size, -1)

        self.c1 = self.c_0_1.expand(batch_size, -1)
        self.c2 = self.c_0_2.expand(batch_size, -1)
        self.c3 = self.c_0_3.expand(batch_size, -1)

        weight_init(self)

    def set_h_c(self, h1, c1, h2, c2):
        self.h1 = h1
        self.c1 = c1
        self.h2 = h2
        self.c2 = c2

    # inputs(length, batch_size, dim)
    def forward(self, inputs, context, char_index):

        if len(inputs.size()) == 1:
            inputs = inputs.type(torch.long).to(device)
            inputs = self.embedding(inputs)
        elif len(inputs.size()) == 0:
            inputs = self.embedding(inputs)
            inputs = torch.unsqueeze(inputs, dim=0)
        else:
            inputs = inputs.squeeze()


        # shoult context view this way?
        context = context.squeeze(1)

        h = torch.cat((inputs, context), 1)


        if char_index == 0:
            self.h1 = self.h_0_1.expand(configuration.batch_size, -1).to(device)
            self.h2 = self.h_0_2.expand(configuration.batch_size, -1).to(device)
            self.h3 = self.h_0_3.expand(configuration.batch_size, -1).to(device)

            self.c1 = self.c_0_1.expand(configuration.batch_size, -1).to(device)
            self.c2 = self.c_0_2.expand(configuration.batch_size, -1).to(device)
            self.c3 = self.c_0_3.expand(configuration.batch_size, -1).to(device)

        self.h1, self.c1 = self.cell1(h, (self.h1, self.c1))

        self.h2, self.c2 = self.cell2(self.h1, (self.h2, self.c2))


        return self.h2, self.c2


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, input):
        metrix, length = pad_packed_sequence(input)
        max_length = metrix.size(0)
        new_length = max_length // 2

        metrix = metrix.transpose(0, 1)

        if max_length % 2 == 1:
            metrix = metrix[:, 0:max_length - 1, :]

        # print(metrix.size())
        metrix = metrix.contiguous().view((-1, new_length, metrix.size(2) * 2))

        metrix = metrix.transpose(0, 1)

        half_length = [l // 2 for l in length]

        returnPack = pack_padded_sequence(metrix, half_length)

        return returnPack


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.bilstm = nn.LSTM(input_size=configuration.frame_dim, hidden_size=configuration.listener_hidden_size,
                              batch_first=False, bidirectional=True)
        self.pbilstm1 = nn.LSTM(input_size=configuration.listener_hidden_size * 4,
                                hidden_size=configuration.listener_hidden_size,
                                batch_first=False, bidirectional=True)
        self.pbilstm2 = nn.LSTM(input_size=configuration.listener_hidden_size * 4,
                                hidden_size=configuration.listener_hidden_size,
                                batch_first=False, bidirectional=True)
        self.pbilstm3 = nn.LSTM(input_size=configuration.listener_hidden_size * 4,
                                hidden_size=configuration.listener_hidden_size,
                                batch_first=False, bidirectional=True)

        self.pooling = Pooling()

        self.key_linear = nn.Linear(in_features=configuration.listener_hidden_size * 2,
                                    out_features=configuration.kqv_size)
        self.value_linear = nn.Linear(in_features=configuration.listener_hidden_size * 2,
                                      out_features=configuration.kqv_size)
        weight_init(self)

    # inputs is a packed sequence, on device
    def forward(self, inputs, length):
        output1, _ = self.bilstm(inputs)

        output2 = self.pooling(output1)

        output3, _ = self.pbilstm1(output2)

        output4 = self.pooling(output3)

        output5, _ = self.pbilstm2(output4)

        output6 = self.pooling(output5)

        output7, _ = self.pbilstm3(output6)

        paded_output7, _ = pad_packed_sequence(output7)

        keys = self.key_linear(paded_output7)

        values = self.value_linear(paded_output7)

        return keys, values


class LasModel(nn.Module):
    def __init__(self, is_train=True):
        super(LasModel, self).__init__()

        self.is_train = is_train

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.linear1 = nn.Linear(in_features=2 * configuration.kqv_size, out_features=configuration.kqv_size)

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(in_features=configuration.kqv_size, out_features=configuration.output_dim)

        self.log_softmax = nn.LogSoftmax()

        weight_init(self)

    def forward(self, x_input, x_length, y_input, y_target, y_length):

        keys, values = self.encoder(x_input, x_length)
        y_input = y_input.squeeze()

        attention_list = []

        fake_attention = torch.FloatTensor(batch_size, 1, values.size(0)).zero_().to(device)
        fake_attention[:, 0, 0:2] = 0.5
        # context
        context = torch.bmm(fake_attention, values.transpose(0, 1))
        # context=torch.FloatTensor(batch_size,1,kqv_size).zero_().to(device)

        max_x_length_after_pbilstm = keys.size(0)

        max_y_length = torch.max(y_length)

        # keys,values,mask shape[batch_size,max_seq,kqv_dim]
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        mask = torch.FloatTensor(batch_size, 1, max_x_length_after_pbilstm).zero_()
        for i in range(x_length.size(0)):
            this_length = int(x_length[i]) // 8
            mask[i, 0, 0:this_length] = torch.ones(this_length).float()
        mask = Variable(mask).to(device)

        out = []

        last_logit = None
        for i in range(max_y_length):

            if self.is_train == True:
                r = random.uniform(0, 1)
                if i == 0:
                    char = torch.LongTensor([32] * batch_size)
                elif r > teacher_forcing:
                    char = y_input[:, i]
                else:
                    predict = self.log_softmax(last_logit)
                    noise = np.random.gumbel(size=predict.size())
                    predict = predict + torch.from_numpy(noise).type(torch.float).to(device)

                    char = torch.bmm(F.softmax(predict / 0.001).unsqueeze(0),
                                     self.decoder.embedding.weight.unsqueeze(0))

            else:
                if i == 0:
                    char = torch.LongTensor([32] * batch_size)
                else:
                    predict = self.log_softmax(last_logit)
                    char = torch.max(predict, dim=1)[1]

            char = char.to(device)
            query, decoder_out = self.decoder(char, context, i)

            energy = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))

            attention = F.softmax(energy, dim=2)

            attention = attention * mask

            attention_list.append(attention[0].squeeze().cpu().detach().numpy())

            attention = attention / torch.sum(attention, 2).unsqueeze(2)

            context = torch.bmm(attention, values)

            context = context.squeeze(1)

            mlp_input = torch.cat((context, query), 1)

            result_1 = self.linear1(mlp_input)
            result_2 = self.relu(result_1)
            result_3 = self.linear2(result_2)

            out += [result_3]
            last_logit = result_3

        out = torch.stack(out, 1)

        stack_attention = np.row_stack(attention_list)

        return out, stack_attention



def weight_init(module):
    if isinstance(module, nn.RNNBase):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.normal_(param.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.normal_(module.bias.data)
