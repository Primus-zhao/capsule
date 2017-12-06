#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F
from ipdb import set_trace
epsilon = 1e-9

# be careful! some pytorch implementation mistakes the dim for squash, we should squash along vec dim instead of channels dim, which means the squash channel for Primary and Digit layer are 8 and 16 respectively
def squash(input, dim):

    norm = torch.norm(input, 2, dim=dim, keepdim=True)
    norm_sq = norm**2
    #s1 = norm_sq/(1.0+norm_sq)*input/norm_sq
    s = norm_sq/(1.0+norm_sq)*input/torch.sqrt(norm_sq + epsilon)

    return s

# some pytorch implmentation mistakes the softmax channels, doesn't like traditional softmax usage, which softmax the layer below, in capsule the digit channel dim is softmaxed, which means that we should softmax dim with value 10
def softmax(input, dim=1):
    vec_length = input.size(dim)
    trans_input = input.transpose(dim, input.size(-1))
    resized_input = trans_input.contiguous().view(-1, vec_length)

    s = F.softmax(resized_input)
    s = s.view(*trans_input.size()).transpose(dim, input.size(-1))

    return s

def one_hot(x, length):
    batch_size = x.size(0)
    x = x.view(batch_size, 1)
    y = torch.LongTensor(x)

    y_onehot = torch.FloatTensor(batch_size, length).zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot
