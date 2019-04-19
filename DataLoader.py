import torch
from torch.utils.data import Dataset
import numpy as np
import configuration
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.rnn as rnn


class WSJDataset(Dataset):

    def __init__(self,trainXpath,trainYpath=None):
        self.yPath=trainYpath

        self.trainX=np.load(trainXpath, encoding='bytes')

        if not trainYpath==None:
            self.trainY = np.load(trainYpath, encoding='bytes')
        else:
            self.trainY=np.random.rand((self.trainX.shape[0]))

        self.trainX=self.trainX[0:10]
        self.trainY=self.trainY[0:10]

        for i in range(self.trainX.shape[0]):
            self.trainX[i]=torch.from_numpy(self.trainX[i])
            if trainYpath!=None:
                self.trainY[i] = torch.from_numpy(self.trainY[i])

    def __getitem__(self, i):
        frameX = self.trainX[i]
        if self.yPath!=None:
            frameY = self.trainY[i]
            return frameX, frameY
        else:
            return frameX,None


    def __len__(self):
        return self.trainY.shape[0]

def collateFrames(frameList):
    xFrame,yFrame=zip(*frameList)
    lens=[len(theFrame) for theFrame in xFrame]
    frameOrder=sorted(range(len(lens)), key=lens.__getitem__,reverse=True)

    xFrame=[xFrame[i] for i in frameOrder]
    yFrame=[yFrame[i] for i in frameOrder]

    yTensor=torch.cat(yFrame)

    xLens = [len(s) for s in xFrame]
    yLens=[len(s) for s in yFrame]

    xBounds = [0]
    for frameLen in xLens:
        xBounds.append(xBounds[-1] + frameLen)


    yBounds = [0]
    for frameLen in yLens:
        yBounds.append(yBounds[-1] + frameLen)

    xLensTensor=torch.tensor(xLens)
    yLensTensor=torch.tensor(yLens)


    return xFrame,yTensor,xBounds,yBounds,xLensTensor,yLensTensor

def collateTest(frameList):
    xFrame, yFrame = zip(*frameList)
    lens = [len(theFrame) for theFrame in xFrame]
    frameOrder = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)

    xFrame = [xFrame[i].float() for i in frameOrder]

    xLens = [len(s) for s in xFrame]

    xBounds = [0]
    for frameLen in xLens:
        xBounds.append(xBounds[-1] + frameLen)

    xLensTensor = torch.tensor(xLens)

    return xFrame,xBounds,xLensTensor


trainXPath=configuration.dataBasePath+'train.npy'
trainYPath=configuration.dataBasePath+'newTrainY.npy'
trainDataSet=WSJDataset(trainXPath,trainYPath)
trainLoader=DataLoader(trainDataSet,shuffle=False,batch_size=5,collate_fn=collateFrames,num_workers=16)


for x,y,xbounds,ybounds,xLens,yLens in trainLoader:
    a=3
