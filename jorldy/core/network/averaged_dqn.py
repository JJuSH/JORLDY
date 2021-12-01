import torch
import torch.nn.functional as F

from .base import BaseNetwork


class Averaged_DQN(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(Averaged_DQN, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.q1 = torch.nn.Linear(D_hidden, D_out)
        self.q2 = torch.nn.Linear(D_hidden, D_out)
        self.q3 = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = super(Averaged_DQN, self).forward(x)
        x1 = self.q1(F.relu(self.l(x)))
        x2 = self.q2(F.relu(self.l(x)))
        x3 = self.q3(F.relu(self.l(x)))
        ret_x = (1/3) * (x1 + x2 + x3)
        #x = F.relu(self.l(x))
        #return self.q(x)
        return ret_x
