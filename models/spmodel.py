import torch.nn as nn
import torch
import numpy as np

class SPNet(nn.Module):
    def __init__(self, patternFile, n):
        super(SPNet, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(n, 28 * 28),  # 32x28x28
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  # 32x28x28
            nn.ReLU(),
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  # 64x14x14
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),  # 64x7x7
        )
        pat = []
        f = open(patternFile, 'rt')
        for line in f.readlines():
            p = [float(x) for x in line.strip().split(' ')]
            pat.append(p)
        self.n = n
        self.mp = torch.FloatTensor(pat[0:n]).t().cuda()
        f.close()

    def forward(self, x):
        batch = x.shape[0]
        c = x.shape[1]
        x1 = x.view(batch, -1)
        x2 = x1.mm(self.mp)
        x2 = x2 * 0.01
        #x3 = nn.Linear(self.n, x.shape[2] * x.shape[3])(x2)
        x3 = self.dense1(x2)
        conv1_out = self.conv1(x3.view(batch, 1, x.shape[2], x.shape[3]))
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        return conv3_out
