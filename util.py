import numpy as np
import configuration
import torch

trainX=np.load(configuration.dataBasePath+"train.npy",encoding="bytes")
trainY=np.load(configuration.dataBasePath+"newTrainY.npy",encoding="bytes")



def get_charmap(corpus):
    chars = list(set(corpus))
    chars.sort()
    charmap = {c: i for i, c in enumerate(chars)}
    charmap['@']=32
    return chars, charmap


def map_corpus(corpus, charmap):
    return torch.IntTensor([int(charmap[c]) for c in corpus])


print(trainY[0].shape)