import numpy as np
import configuration
import torch
import matplotlib.pyplot as plt
import time

# trainX=np.load(configuration.dataBasePath+"train.npy",encoding="bytes")
# trainY=np.load(configuration.dataBasePath+"newTrainY.npy",encoding="bytes")



def get_charmap(corpus):
    chars = list(set(corpus))
    chars.sort()
    charmap = {c: i for i, c in enumerate(chars)}
    charmap['@']=32
    charmap['?'] =33
    return chars, charmap


def map_corpus(corpus, charmap):
    return torch.IntTensor([int(charmap[c]) for c in corpus])


# print(trainY[0].shape)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=10)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


def show_attention_weights(attention_weights):
    fig = plt.figure()
    plt.imshow(attention_weights,cmap='hot')
    fig.savefig("./graph/%d.png" % (time.time()))
    # fig.show()
    plt.close()