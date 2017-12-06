import torch
import torch.nn as nn
from ipdb import set_trace

# this TimeDistributed class just wants to mimic its counterpart in keras, since pytorch doens't provide this fcn right now, if you don't like it, you can use pytorch ModuleList to realize the same fcn in capsnet layers, it's quite easy 
class TimeDistributed(nn.Module):
    def __init__(self, module, time_steps = 1):
        super(TimeDistributed, self).__init__()
        self.time_steps = time_steps
        self.module = []
        for i in range(self.time_steps):
            self.module.append(module)

    def forward(self, x):

        tup = torch.split(x, 1, dim=1)
        assert len(tup)==self.time_steps,'data time steps not right!'

        y = []
        for i in range(self.time_steps):
            x_slice = tup[i].squeeze(1)  
            y.append(self.module[i](x_slice))

        result = torch.stack(y, dim=1)

        return result
