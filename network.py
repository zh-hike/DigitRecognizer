import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.4):
        super(Block, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p)
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            Block(784, 512),
            Block(512, 256),
            Block(256, 128),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.net(x)