#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import softmax, squash
from time_distributed import TimeDistributed
from ipdb import set_trace

class Capsule(nn.Module):
    def __init__(self, input_height, input_width, input_channels, conv_channels, primary_channels, digit_channels, primary_capsule_num, primary_vec_length, digit_vec_length):
        super(Capsule, self).__init__()
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.primary_channels = primary_channels 
        self.digit_channels = digit_channels
        self.primary_capsule_num = primary_capsule_num 
        self.primary_vec_length = primary_vec_length
        self.digit_vec_length = digit_vec_length

        self.conv = ConvLayer(in_channels = self.input_channels,
                              out_channels = self.conv_channels)

        self.primary = PrimaryLayer(conv_channels = self.conv_channels, 
                                    primary_channels = self.primary_channels, 
                                    primary_vec_length = self.primary_vec_length)

        self.digit = DigitLayer(self.primary_channels, self.digit_channels, self.primary_capsule_num, self.primary_vec_length, self.digit_vec_length)

        input_size = input_height * input_width * input_channels
        self.recon0 = nn.Linear(digit_channels * digit_vec_length, input_size/3)
        self.recon1 = nn.Linear(input_size/3, input_size/2)
        self.recon2 = nn.Linear(input_size/2, input_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return self.digit(self.primary(self.conv(x)))

    def loss(self, output, label, original_images):
        m_loss = self.margin_loss(output, label)
        r_loss = self.recon_loss(output, original_images)

        return self.margin_loss(output, label) + 0.0005*self.recon_loss(output, original_images)

    def margin_loss(self, x, label, batch_average = True):
        # x -> [batch_size, digit_channels, digit_vec_length, 1]
        batch_size = x.size(0)
        
        # x_mag -> [batch_size, digit_channels, 1, 1]
        x_mag = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        m_plus = 0.9
        m_minus = 0.1
        zero = Variable(torch.zeros(1).cuda())

        loss_correct = torch.max((m_plus - x_mag), zero).squeeze()
        loss_false = torch.max((x_mag - m_minus), zero).squeeze()

        loss_lambda = 0.5
        T_c = label
        one = Variable(torch.ones(1)).cuda()
        L_c = T_c * loss_correct**2 + loss_lambda * (1.0 - T_c) * loss_false**2
        L_c = L_c.sum(dim=1)
        
        if batch_average: L_c = L_c.mean()

        return L_c

    def recon_loss(self, x, original_images, batch_average = True):
        # x -> [batch_size, digit_channels, digit_vec_length, 1]
        # x_mag -> [batch_size, digit_channels, 1, 1]
        batch_size = x.size(0)
        x_mag = torch.sqrt((x**2).sum(dim=2))
        val, max_ind = x_mag.max(dim = 1)
        max_ind = max_ind.squeeze()

        masked_x = Variable(torch.zeros(*x.size())).cuda()

        for batch in xrange(batch_size):
            #masked false prediction
            masked_x[batch, max_ind[batch].data[0], :, :] = x[batch, max_ind[batch].data[0], :, :]
        
        masked_x = masked_x.view(batch_size, -1)
        recon = self.relu(self.recon0(masked_x))
        recon = self.relu(self.recon1(recon))
        recon = self.sigmoid(self.recon2(recon))

        error = (recon - original_images.view(batch_size, -1))
        error = error**2

        error = error.sum(dim=1)

        if batch_average == True: error = error.mean()

        return error

        


class ConvLayer(nn.Module):
    '''this is the first convolution layer of capsnet'''
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv_fcn = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=9,
                                  stride=1,
                                  bias=True)
        self.relu_fcn = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_fcn(self.conv_fcn(x))

class PrimaryLayer(nn.Module):
    def __init__(self, conv_channels, primary_channels, primary_vec_length=8):
        super(PrimaryLayer, self).__init__()
        self.primary_channels = primary_channels
        self.primary_vec_length = primary_vec_length
        self.conv_fcn = nn.Conv2d(conv_channels,
                                 primary_channels,
                                 kernel_size=9,
                                 stride=2,
                                 bias=True)
        self.distributed_conv_fcn = TimeDistributed(self.conv_fcn, time_steps=self.primary_vec_length)

    def forward(self, x):
        """it's necessary to mention that the output of convolution can be relued before sending to next layer, in capsnet paper, nothing has mentioned about this, so we just remind you here that it's a possible option, in the tensorflow implementation of capsnet written by naturomics, relu is used"""
        batch_size = x.size(0)
        conv = self.distributed_conv_fcn(torch.stack([x] * self.primary_vec_length, dim=1))

        # squ -> [batch, primary_capsule_num, primary_vec_length, 1]
        squ = torch.transpose(squash(conv, dim=1).view(batch_size, self.primary_vec_length, -1), 1, 2)
        squ = squ.unsqueeze(3)

        return squ

class DigitLayer(nn.Module):
    def __init__(self, primary_channels, digit_channels, primary_capsule_num,  primary_vec_length=8, digit_vec_length=16):
        super(DigitLayer, self).__init__()
        self.primary_capsule_num = primary_capsule_num
        self.primary_channels = primary_channels
        self.digit_channels = digit_channels
        self.primary_vec_length = primary_vec_length
        self.digit_vec_length = digit_vec_length
        # W is fixed and updated by backpro, while b_ij is reset to 0 every time and updated by dynamic route, we also need to mention that in pytorch, randn function has default stddev 1, which is too large for this implementation, so we divide 100.0 to squeeze it to 0.01, it's important for stable and fast training
        self.W = nn.Parameter(torch.randn(primary_capsule_num, digit_channels, digit_vec_length, primary_vec_length)/100.0).cuda()

    def forward(self, x):
        batch_size = x.size(0)
        # x -> [batch_size, primary_capsule_num,digit_channels primary_vec_length, 1]
        x = torch.stack([x] * self.digit_channels, dim=2)
        # W -> [batch_size, primary_capsule_num, digit_channels, digit_vec_length, primary_vec_length]
        W = torch.stack([self.W]*batch_size, dim=0)
        #b_ij will be reset everytime forwarded
        b_ij = Variable(torch.zeros(batch_size, self.primary_capsule_num, self.digit_channels, 1, 1)).cuda()

        # u_hat -> [batch_size, primary_capsule_num, digit_channels, digit_vec_length, 1]
        u_hat = torch.matmul(W, x)
        #in dynamic routing, b_ij will be updated 3 times, in the first two times we just calculate v_ij for update b_ij, the result shouldn't influence other weights since b_ij is still in unstable state, which means that backpro shouldn't be executed for these two iterations, so we use u_hat_stopped to detach gradient backpro, several implementation ignore this
        u_hat_stopped = u_hat.detach()

        num_iterations = 3
        for iteration in range(num_iterations):
            # c_ij -> [batch_size, primary_capsule_num, digit_channels, 1, 1]
            c_ij = softmax(b_ij, dim=3)

            # s_j -> [batch_size, digit_channels, digit_vec_length, 1]
            if iteration == num_iterations-1:
                s_j = (c_ij * u_hat).sum(dim=1)
            else:
                s_j = (c_ij * u_hat_stopped).sum(dim=1)
            
            # v_j -> [batch_size, digit_channels, digit_vec_length, 1]
            v_j = squash(s_j, 2)

            if iteration != num_iterations -1:
                # update -> [batch_size, primary_capsule_num, digit_channels, 1, 1]
                update = torch.matmul(torch.transpose(u_hat, -1, -2) , torch.stack([v_j]*self.primary_capsule_num, dim=1))

                b_ij = b_ij + update

        return v_j

