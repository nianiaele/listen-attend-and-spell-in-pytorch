import numpy as np
import configuration
import util
from Dictionary import charDic

# print(configuration.dataBasePath)
# trainX=np.load(configuration.dataBasePath+"train.npy",encoding="bytes")
# trainY=np.load(configuration.dataBasePath+"train_transcripts.npy",encoding="bytes")
#
# devY=np.load(configuration.dataBasePath+"dev_transcripts.npy",encoding="bytes")
#
# append_count=[]
# for i in range(trainY.shape[0]):
#     a = [string.decode() for string in trainY[i]]
#     append_count=append_count+a
# corpus = " ".join(append_count)
# chars, charmap = util.get_charmap(corpus)
#
#
# dictionary_length = len(chars)
# print("Unique character count: {}".format(len(chars)))
# array = util.map_corpus(corpus, charDic)
#
# print(array)

# newTrianY=trainY.copy()
#
# for i in range(trainY.shape[0]):
#     a = [string.decode() for string in trainY[i]]
#     b=" ".join(a)
#     b='@'+b+"@"
#     c=util.map_corpus(b,charmap)
#     d=c.numpy()
#     e=d
#
#     newTrianY[i]=e
#
# np.save(configuration.dataBasePath+"newTrainY.npy",newTrianY)
#
# newDevY=devY.copy()
#
# for i in range(devY.shape[0]):
#     a = [string.decode() for string in devY[i]]
#     b=" ".join(a)
#     b='@'+b+"@"
#     c=util.map_corpus(b,charmap)
#     d=c.numpy()
#     e=d
#
#     newDevY[i]=e
#
# np.save(configuration.dataBasePath+"newDevY.npy",newDevY)

import random
r=random.uniform(0,1)
print("number",r)