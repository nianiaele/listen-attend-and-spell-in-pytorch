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
from Dictionary import char_List
class beam_node():
    def __init__(self,beam_width):
        self.history_list=[]
        self.current_value=1
        self.context=None

    def copy(self,old):
        self.current_value=old.current_value
        self.context=old.context

        for k in old.history_list:
            self.history_list.append(k)

    # def __cmp__(self, other):
    #     return self.current_value > other.current_value
    def __lt__(self, other):
        return self.current_value > other.current_value


model=LasModel()

print("loading model")
checkPoint = torch.load(dataBasePath+"myModel", map_location='cpu')

model.load_state_dict(checkPoint['model_state_dict'])


dev_x_path=dataBasePath+"test.npy"
dev_dataset=WSJDataset(dev_x_path)
test_loader=DataLoader(dev_dataset,shuffle=False,batch_size=1,collate_fn=collateTest,num_workers=2)

def beam_search(model,data_loader,beam_width):


    encoder=model.encoder
    decoder=model.decoder

    linear1=model.linear1
    linear2=model.linear2
    relu=model.relu

    result=[]

    model.is_train=False

    beam_list=[0 for i in range(beam_width)]



    for i in range(beam_width):
        beam_list[i]=beam_node(beam_width)

    for x, xbounds, xLens in test_loader:

        x=pack_sequence(x)
        keys, values = encoder(x, xLens)


        context = Variable(torch.FloatTensor(1, 1, kqv_size).zero_()).to(device)

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

        beam_queue = queue.PriorityQueue(2 * beam_width)
        history_list=one_seq_beam(decoder,keys,mask,values,linear1,linear2,relu,beam_queue,context)
        print(history_list)
        result.append(history_list)
    return result

def one_seq_beam(decoder,keys,mask,values,linear1,linear2,relu,beam_queue,context):
    ii=0
    softmax=nn.Softmax(dim=2)
    while True:
        if ii == 0:
            char = torch.LongTensor([32] * 1)

            query,decoder_out = decoder(char, context, ii)

            energy = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))

            attention = softmax(energy)

            attention = attention * mask

            attention = attention / torch.sum(attention, 2).unsqueeze(2)

            context = torch.bmm(attention, values)

            context = context.squeeze(1)

            mlp_input = torch.cat((context, decoder_out), 1)

            logit = linear1(mlp_input)
            logit = relu(logit)
            logit = linear2(logit)

            probability = F.softmax(logit, dim=1).view(-1)

            sorted, indice = torch.sort(probability, descending=True)

            for j in range(beam_width):

                node = beam_node(j)

                node.history_list.append(indice[j].item())
                # node.currentValue = beam_list[i].current_value * probability[pos]
                node.context = context

                node.current_value=node.current_value*sorted[j].item()

                beam_queue.put(node)
        else:

            for j in range(min(beam_width, beam_queue.qsize())):
                node = beam_queue.get()
                char = node.history_list[-1]

                context = node.context

                input = torch.tensor(char)
                query,decoder_out = decoder(input.view(input.size(), 1), context, ii)

                energy = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))

                attention = F.softmax(energy, dim=2)

                attention = attention * mask

                attention = attention / torch.sum(attention, 2).unsqueeze(2)

                context = torch.bmm(attention, values)

                context = context.squeeze(1)

                mlp_input = torch.cat((context, decoder_out), 1)

                logit = linear1(mlp_input)
                logit = relu(logit)
                logit = linear2(logit)

                probability = F.softmax(logit, dim=1).view(-1)

                sorted, indice = torch.sort(probability, descending=True)


                for j in range(beam_width):
                    pos = indice[j].item()

                    new_node = beam_node(j)
                    new_node.copy(node)

                    new_node.history_list.append(indice[j].item())
                    new_node.current_value = node.current_value * probability[pos].item()
                    new_node.context = context

                    if indice[j].item() == 32:
                        return new_node.history_list

                    if len(new_node.history_list) > max_generate_length:
                        return new_node.history_list

                    try:
                        beam_queue.put(new_node, block=False)
                    except:
                        new_beam_queue = queue.PriorityQueue(2 * beam_width)
                        for k in range(beam_width * 2 - 1):
                            new_beam_queue.put(beam_queue.get())
                        beam_queue = new_beam_queue

                        beam_queue.put(new_node, block=False)

        ii+=1


def indexToChar(index_list):
    string=""
    for i in index_list:
        string+=char_List[i]
    return string



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