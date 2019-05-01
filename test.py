import numpy as np
import configuration
import util
from Dictionary import charDic
import torch
import torch.nn.functional as F
from Dictionary import charDic


# print(configuration.dataBasePath)
trainX=np.load(configuration.dataBasePath+"train.npy",encoding="bytes",allow_pickle=True)
trainY=np.load(configuration.dataBasePath+"train_transcripts.npy",encoding="bytes",allow_pickle=True)

devX=np.load(configuration.dataBasePath+"dev.npy",encoding="bytes",allow_pickle=True)
devY=np.load(configuration.dataBasePath+"dev_transcripts.npy",encoding="bytes",allow_pickle=True)

# append_count=[]
# for i in range(trainY.shape[0]):
#     a = [string.decode() for string in trainY[i]]
#     append_count=append_count+a
# corpus = " ".join(append_count)
# chars, charmap = util.get_charmap(corpus)


# dictionary_length = len(chars)
# print("Unique character count: {}".format(len(chars)))
# array = util.map_corpus(corpus, charDic)
#
# print(array)

newTrianY=trainY.copy()

for i in range(trainY.shape[0]):
    a = [string.decode() for string in trainY[i]]
    b=" ".join(a)
    b='@'+b+"?"
    c=util.map_corpus(b,charDic)
    d=c.numpy()
    e=d

    print(i)

    newTrianY[i]=e

np.save(configuration.dataBasePath+"newTrainY.npy",newTrianY,allow_pickle=True)

newDevY=devY.copy()

for i in range(devY.shape[0]):
    a = [string.decode() for string in devY[i]]
    b=" ".join(a)
    b='@'+b+"?"
    c=util.map_corpus(b,charDic)
    d=c.numpy()
    e=d

    print(i)

    newDevY[i]=e

np.save(configuration.dataBasePath+"newDevY.npy",newDevY,allow_pickle=True)

m = torch.nn.Softmax(dim=0)
input = torch.tensor([0.2,0.3])
output = m(input.type(torch.float))
print(output)


