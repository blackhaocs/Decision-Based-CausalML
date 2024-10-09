import torch.nn as nn
import torch

class slearner_criteo_uplift(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mlp = nn.Sequential(nn.Linear(input_size, 127),
                                  nn.ReLU())
        self.header1 = nn.Sequential(nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.header2 = nn.Sequential(nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.header3 = nn.Sequential(nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.header4 = nn.Sequential(nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, T0, T1):
        h = self.mlp(x)
        output1 = self.sigmoid(self.header1(torch.cat([h, T0], dim=1)))
        output2 = self.sigmoid(self.header2(torch.cat([h, T1], dim=1)))
        output3 = self.sigmoid(self.header3(torch.cat([h, T0], dim=1)))
        output4 = self.sigmoid(self.header4(torch.cat([h, T1], dim=1)))
        return output1, output2, output3, output4