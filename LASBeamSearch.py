import torch
from Model import LasModel
from DataLoader import WSJDataset,collateTest
from configuration import dataBasePath,predict_result,batch_size,kqv_size,device,max_generate_length,beam_width
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pack_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import queue
import csv
from Dictionary import char_List,to_string
import math


class beam_node():
    def __init__(self):
        self.history_list=[]
        self.current_value=1
        self.context=None
        self.score=0

    def set_h_c(self,h1,c1,h2,c2):
        self.h1=h1
        self.c1=c1
        self.h2=h2
        self.c2=c2

    def get_h_c(self):
        return self.h1,self.c1,self.h2,self.c2

    def copy(self,old):
        self.current_value=old.current_value
        self.context=old.context

        for k in old.history_list:
            self.history_list.append(k)

    def get_score(self):
        self.score=math.log(self.current_value)/len(self.history_list)

    # def __lt__(self, other):
    #     return self.current_value > other.current_value
    def __lt__(self, other):
        return self.score > other.score


model=LasModel()

print("loading model")
checkPoint = torch.load("./myModel", map_location='cpu')

model.load_state_dict(checkPoint['model_state_dict'])


dev_x_path=dataBasePath+"test.npy"
dev_dataset=WSJDataset(dev_x_path)
test_loader=DataLoader(dev_dataset,shuffle=False,batch_size=1,collate_fn=collateTest,num_workers=8)

def beam_search(model,data_loader,beam_width):

    encoder=model.encoder
    decoder=model.decoder

    linear1=model.linear1
    linear2=model.linear2
    relu=model.relu

    result=[]

    model.is_train=False



    # for i in range(beam_width):
    #     beam_list[i]=beam_node()

    for x, xbounds, xLens in data_loader:

        x=pack_sequence(x)
        keys, values = encoder(x, xLens)

        fake_attention=torch.FloatTensor(1,1,values.size(0)).zero_().to(device)
        fake_attention[:,0,0:2]=0.5
        context=torch.bmm(fake_attention,values.transpose(0,1))

        max_x_length_after_pbilstm=keys.size(0)


        #keys,values,mask shape[batch_size,max_seq,kqv_dim]
        keys=keys.transpose(0,1)
        values=values.transpose(0,1)

        mask=torch.FloatTensor(1,1,max_x_length_after_pbilstm).zero_()
        for i in range(xLens.size(0)):
            this_length=int(xLens[i])//8
            mask[i,0,0:this_length]=torch.ones(this_length).float()
        mask=Variable(mask).to(device)

        got_eos=False
        i=0

        beam_queue = queue.PriorityQueue(3 * beam_width)

        result_list=one_seq_beam(decoder,keys,mask,values,linear1,linear2,relu,beam_queue,context)

        # for nodee in result_list:
        #     print(indexToChar(nodee.history_list)+ str(nodee.current_value))




        best_node=find_best_result(result_list)

        history_list=best_node.history_list[:-1]

        print(indexToChar(history_list))

        # print_result(result_list)
        # exit(0)

        result.append(history_list)
    return result



def one_seq_beam(decoder,keys,mask,values,linear1,linear2,relu,beam_queue,context):
    ii=0
    softmax=nn.Softmax(dim=2)
    result_list=[]
    while True:
        if ii == 0:
            char = torch.LongTensor([32] * 1)

            query,c = decoder(char, context, ii)

            energy = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))

            attention = softmax(energy)

            attention = attention * mask

            attention = attention / torch.sum(attention, 2).unsqueeze(2)

            context = torch.bmm(attention, values)

            context = context.squeeze(1)

            mlp_input = torch.cat((context, query), 1)

            logit = linear1(mlp_input)
            logit = relu(logit)
            logit = linear2(logit)

            probability = F.softmax(logit, dim=1).view(-1)

            sorted, indice = torch.sort(probability, descending=True)

            for j in range(len(char_List)):

                node = beam_node()

                node.set_h_c(decoder.h1,decoder.c1,decoder.h2,decoder.c2)

                node.history_list.append(indice[j].item())
                # node.currentValue = beam_list[i].current_value * probability[pos]
                node.context = context

                node.current_value=node.current_value*sorted[j].item()

                node.get_score()

                beam_queue.put(node)
        else:

            node = beam_queue.get()
            char = node.history_list[-1]

            context = node.context

            input = torch.tensor(char)


            h1,c1,h2,c2=node.get_h_c()
            decoder.set_h_c(h1,c1,h2,c2)

            query, decoder_out = decoder(input.view(input.size(), 1), context, ii)

            energy = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))

            attention = F.softmax(energy, dim=2)

            attention = attention * mask

            attention = attention / torch.sum(attention, 2).unsqueeze(2)

            context = torch.bmm(attention, values)

            context = context.squeeze(1)

            mlp_input = torch.cat((context, query), 1)

            logit = linear1(mlp_input)
            logit = relu(logit)
            logit = linear2(logit)

            probability = F.softmax(logit, dim=1).view(-1)

            sorted, indice = torch.sort(probability, descending=True)

            for j in range(10):
                pos = indice[j].item()

                new_node = beam_node()
                new_node.set_h_c(decoder.h1,decoder.c1,decoder.h2,decoder.c2)
                new_node.copy(node)

                new_node.history_list.append(indice[j].item())
                new_node.current_value = node.current_value * probability[pos].item()
                new_node.context = context
                new_node.get_score()

                if indice[j].item() == 33:
                    if len(new_node.history_list)>10:
                        result_list.append(new_node)
                    continue

                if len(result_list)>5*beam_width:
                    return result_list

                # print(len(new_node.history_list))
                # if len(new_node.history_list)>max_generate_length:
                #     return result_list

                try:
                    beam_queue.put(new_node, block=False)
                except:
                    new_beam_queue = queue.PriorityQueue(3 * beam_width)
                    for k in range(beam_width * 3 - 1):
                        new_beam_queue.put(beam_queue.get())
                    beam_queue = new_beam_queue

                    beam_queue.put(new_node, block=False)

        ii+=1

    return result_list


def indexToChar(index_list):
    string=""
    for i in index_list:
        string+=char_List[i]
    return string

def print_result(result_list):
    for node_i in range(len(result_list)):
        print(result_list[node_i].score)
        print(indexToChar(result_list[node_i].history_list))
        print("-----------------------------------------------------")


def find_best_result(result_list):
    best_node=result_list[0]
    for node_i in range(1,len(result_list)):
        curr_node=result_list[node_i]
        if curr_node.score>best_node.score:
            best_node=curr_node
    return best_node

# model=LasModel()
a=beam_search(model,test_loader,beam_width=beam_width)

with open("./submission.csv", mode='w', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(("Id","Predicted"))
    for i in range(len(a)):
        string=indexToChar(a[i])
        writer.writerow((i,string))


# i need to recover the value to reduce the effect of different lengh
# keep all sequence and retune the largest one