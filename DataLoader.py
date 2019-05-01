import torch
from torch.utils.data import Dataset
import numpy as np
import configuration
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence


class WSJDataset(Dataset):

    def __init__(self, trainXpath, trainYpath=None):
        self.yPath = trainYpath

        self.trainX = np.load(trainXpath, encoding='bytes',allow_pickle=True)

        if not trainYpath == None:
            self.trainY = np.load(trainYpath, encoding='bytes',allow_pickle=True)
        else:
            self.trainY = np.random.rand((self.trainX.shape[0]))

        self.trainX = self.trainX
        self.trainY = self.trainY

        for i in range(self.trainX.shape[0]):
            self.trainX[i] = torch.from_numpy(self.trainX[i])
            if trainYpath != None:
                self.trainY[i] = torch.from_numpy(self.trainY[i])

    def __getitem__(self, i):
        frameX = self.trainX[i]
        if self.yPath != None:
            frameY = self.trainY[i]
            return frameX, frameY
        else:
            return frameX, None

    def __len__(self):
        return self.trainY.shape[0]


def collateFrames(frameList):
    xFrame, yFrame = zip(*frameList)
    lens = [len(theFrame) for theFrame in xFrame]
    frameOrder = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)

    xFrame = [xFrame[i] for i in frameOrder]
    yFrame = [yFrame[i] for i in frameOrder]

    # pad_y=pad_sequence(xFrame)

    inputY = [yFrame[i][0:yFrame[i].shape[0] - 1] for i in range(len(frameOrder))]
    targetY = [yFrame[i][1:yFrame[i].shape[0]] for i in range(len(frameOrder))]

    padded_inputY = pad_sequence(inputY).transpose(0,1)
    padded_targetY = pad_sequence(targetY).transpose(0,1)



    yTensor = torch.cat(yFrame)

    xLens = [len(s) for s in xFrame]
    yLens = [len(s) - 1 for s in yFrame]

    xBounds = [0]
    for frameLen in xLens:
        xBounds.append(xBounds[-1] + frameLen)

    yBounds = [0]
    for frameLen in yLens:
        yBounds.append(yBounds[-1] + frameLen)

    xLensTensor = torch.tensor(xLens)
    yLensTensor = torch.tensor(yLens)



    return xFrame, yTensor, xBounds, yBounds, xLensTensor, yLensTensor, padded_inputY, padded_targetY


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

    return xFrame, xBounds, xLensTensor


# trainXPath = configuration.dataBasePath + 'train.npy'
# trainYPath = configuration.dataBasePath + 'newTrainY.npy'
# trainDataSet = WSJDataset(trainXPath, trainYPath)
# trainLoader = DataLoader(trainDataSet, shuffle=False, batch_size=5, collate_fn=collateFrames, num_workers=16)

# devX=np.load(configuration.dataBasePath+"dev.npy",encoding="bytes")
# devY=np.load(configuration.dataBasePath+"dev_transcripts.npy",encoding="bytes")
#
#
# dev_x_path=configuration.dataBasePath+"dev.npy"
# dev_dataset=WSJDataset(dev_x_path)
# test_loader=DataLoader(dev_dataset,shuffle=False,batch_size=2,collate_fn=collateTest,num_workers=2)
#
# for x,xbounds,xLens in test_loader:
#     aa=4
