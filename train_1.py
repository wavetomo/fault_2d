import torch.nn as nn
import torch
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import os
from os import listdir
import scipy
from os.path import splitext
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
# from .models.networks_other import Unetconv_norm_lrelu, Unetnorm_lrelu_conv,Unetlrelu_conv,Unetnorm_lrelu_upscale_conv_norm_lrelu
from torch.nn import init
import torch
import math
import torch.nn as nn
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
import torch.nn as nn
import torch
from torchsummary import summary
from glob import glob
from os import listdir
from os.path import splitext
import torch.utils.data
from pytorchtools import EarlyStopping
import cmapy
from functions import *
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# ===============================#
#         初始化权重              #
# ===============================#


def weights_init(net, init_type='kaiming', init_gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and classname.find('Conv') != -1:
			if init_type == 'normal':
				torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				torch.nn.init.kaiming_normal_(
					m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError(
					'initialization method [%s] is not implemented' % init_type)
		elif classname.find('BatchNorm2d') != -1:
			torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
			torch.nn.init.constant_(m.bias.data, 0.0)

	print('initialize network with %s type' % init_type)
	net.apply(init_func)


# =================pidinet==========#


class Conv2d(nn.Module):
	def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
				 bias=False):
		super(Conv2d, self).__init__()
		if in_channels % groups != 0:
			raise ValueError('in_channels must be divisible by groups')
		if out_channels % groups != 0:
			raise ValueError('out_channels must be divisible by groups')
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.weight = nn.Parameter(torch.Tensor(
			out_channels, in_channels // groups, kernel_size, kernel_size))
		# print(self.weight.shape)
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		self.pdc = pdc

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, input):

		return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# cd, ad, rd convolutions
def createConvFunc(op_type):
	assert op_type in ['cv', 'cd', 'ad',
					   'rd'], 'unknown op type: %s' % str(op_type)
	if op_type == 'cv':
		return F.conv2d

	if op_type == 'cd':
		def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
			assert dilation in [
				1, 2], 'dilation for cd_conv should be in 1 or 2'
			assert weights.size(2) == 3 and weights.size(
				3) == 3, 'kernel size for cd_conv should be 3x3'
			assert padding == dilation, 'padding for cd_conv set wrong'

			weights_c = weights.sum(dim=[2, 3], keepdim=True)
			yc = F.conv2d(x, weights_c, stride=stride,
						  padding=0, groups=groups)
			y = F.conv2d(x, weights, bias, stride=stride,
						 padding=padding, dilation=dilation, groups=groups)
			return y - yc

		return func
	elif op_type == 'ad':
		def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
			assert dilation in [
				1, 2], 'dilation for ad_conv should be in 1 or 2'
			assert weights.size(2) == 3 and weights.size(
				3) == 3, 'kernel size for ad_conv should be 3x3'
			assert padding == dilation, 'padding for ad_conv set wrong'

			shape = weights.shape
			weights = weights.view(shape[0], shape[1], -1)
			# print("weights_c.shape:{}".format(weights.shape))
			weights_conv = (
					weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
			y = F.conv2d(x, weights_conv, bias, stride=stride,
						 padding=padding, dilation=dilation, groups=groups)
			return y

		return func
	elif op_type == 'rd':
		def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
			assert dilation in [
				1, 2], 'dilation for rd_conv should be in 1 or 2'
			assert weights.size(2) == 3 and weights.size(
				3) == 3, 'kernel size for rd_conv should be 3x3'
			padding = 2 * dilation

			shape = weights.shape
			if weights.is_cuda:
				buffer = torch.cuda.FloatTensor(
					shape[0], shape[1], 5 * 5).fill_(0)
			else:
				buffer = torch.zeros(shape[0], shape[1], 5 * 5)
			weights = weights.view(shape[0], shape[1], -1)
			buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
			buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
			buffer[:, :, 12] = 0
			buffer = buffer.view(shape[0], shape[1], 5, 5)
			y = F.conv2d(x, buffer, bias, stride=stride,
						 padding=padding, dilation=dilation, groups=groups)
			return y

		return func
	else:
		print('impossible to be here unless you force that')
		return None


nets = {
	'baseline': {
		'layer0': 'cv',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'cv',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'cv',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'cv',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'c-v15': {
		'layer0': 'cd',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'cv',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'cv',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'cv',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'a-v15': {
		'layer0': 'ad',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'cv',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'cv',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'cv',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'r-v15': {
		'layer0': 'rd',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'cv',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'cv',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'cv',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'cvvv4': {
		'layer0': 'cd',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'cd',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'cd',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'cd',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'avvv4': {
		'layer0': 'ad',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'ad',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'ad',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'ad',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'rvvv4': {
		'layer0': 'rd',
		'layer1': 'cv',
		'layer2': 'cv',
		'layer3': 'cv',
		'layer4': 'rd',
		'layer5': 'cv',
		'layer6': 'cv',
		'layer7': 'cv',
		'layer8': 'rd',
		'layer9': 'cv',
		'layer10': 'cv',
		'layer11': 'cv',
		'layer12': 'rd',
		'layer13': 'cv',
		'layer14': 'cv',
		'layer15': 'cv',
	},
	'cccv4': {
		'layer0': 'cd',
		'layer1': 'cd',
		'layer2': 'cd',
		'layer3': 'cv',
		'layer4': 'cd',
		'layer5': 'cd',
		'layer6': 'cd',
		'layer7': 'cv',
		'layer8': 'cd',
		'layer9': 'cd',
		'layer10': 'cd',
		'layer11': 'cv',
		'layer12': 'cd',
		'layer13': 'cd',
		'layer14': 'cd',
		'layer15': 'cv',
	},
	'aaav4': {
		'layer0': 'ad',
		'layer1': 'ad',
		'layer2': 'ad',
		'layer3': 'cv',
		'layer4': 'ad',
		'layer5': 'ad',
		'layer6': 'ad',
		'layer7': 'cv',
		'layer8': 'ad',
		'layer9': 'ad',
		'layer10': 'ad',
		'layer11': 'cv',
		'layer12': 'ad',
		'layer13': 'ad',
		'layer14': 'ad',
		'layer15': 'cv',
	},
	'rrrv4': {
		'layer0': 'rd',
		'layer1': 'rd',
		'layer2': 'rd',
		'layer3': 'cv',
		'layer4': 'rd',
		'layer5': 'rd',
		'layer6': 'rd',
		'layer7': 'cv',
		'layer8': 'rd',
		'layer9': 'rd',
		'layer10': 'rd',
		'layer11': 'cv',
		'layer12': 'rd',
		'layer13': 'rd',
		'layer14': 'rd',
		'layer15': 'cv',
	},
	'c16': {
		'layer0': 'cd',
		'layer1': 'cd',
		'layer2': 'cd',
		'layer3': 'cd',
		'layer4': 'cd',
		'layer5': 'cd',
		'layer6': 'cd',
		'layer7': 'cd',
		'layer8': 'cd',
		'layer9': 'cd',
		'layer10': 'cd',
		'layer11': 'cd',
		'layer12': 'cd',
		'layer13': 'cd',
		'layer14': 'cd',
		'layer15': 'cd',
	},
	'a16': {
		'layer0': 'ad',
		'layer1': 'ad',
		'layer2': 'ad',
		'layer3': 'ad',
		'layer4': 'ad',
		'layer5': 'ad',
		'layer6': 'ad',
		'layer7': 'ad',
		'layer8': 'ad',
		'layer9': 'ad',
		'layer10': 'ad',
		'layer11': 'ad',
		'layer12': 'ad',
		'layer13': 'ad',
		'layer14': 'ad',
		'layer15': 'ad',
	},
	'r16': {
		'layer0': 'rd',
		'layer1': 'rd',
		'layer2': 'rd',
		'layer3': 'rd',
		'layer4': 'rd',
		'layer5': 'rd',
		'layer6': 'rd',
		'layer7': 'rd',
		'layer8': 'rd',
		'layer9': 'rd',
		'layer10': 'rd',
		'layer11': 'rd',
		'layer12': 'rd',
		'layer13': 'rd',
		'layer14': 'rd',
		'layer15': 'rd',
	},
	'carv4': {
		'layer0': 'cd',
		'layer1': 'ad',
		'layer2': 'rd',
		'layer3': 'cv',
		'layer4': 'cd',
		'layer5': 'ad',
		'layer6': 'rd',
		'layer7': 'cv',
		'layer8': 'cd',
		'layer9': 'ad',
		'layer10': 'rd',
		'layer11': 'cv',
		'layer12': 'cd',
		'layer13': 'ad',
		'layer14': 'rd',
		'layer15': 'cv',
	},
}


def config_model(model):
	model_options = list(nets.keys())
	assert model in model_options, \
		'unrecognized model, please choose from %s' % str(model_options)

	print(str(nets[model]))

	pdcs = []
	for i in range(16):
		layer_name = 'layer%d' % i
		op = nets[model][layer_name]
		pdcs.append(createConvFunc(op))

	return pdcs


def config_model_converted(model):
	model_options = list(nets.keys())
	assert model in model_options, \
		'unrecognized model, please choose from %s' % str(model_options)

	print(str(nets[model]))

	pdcs = []
	for i in range(16):
		layer_name = 'layer%d' % i
		op = nets[model][layer_name]
		pdcs.append(op)

	return pdcs


# class CSAM(nn.Module):
#     """
#     Compact Spatial Attention Module
#     """

#     def __init__(self, channels):
#         super(CSAM, self).__init__()

#         mid_channels = 4
#         self.relu1 = nn.ReLU()
#         self.conv1 = nn.Conv2d(channels, mid_channels,
#                                kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(
#             mid_channels, 1, kernel_size=3, padding=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         nn.init.constant_(self.conv1.bias, 0)
#         self.dropout=nn.Dropout(0.2)
#     def forward(self, x):
#         y = self.relu1(x)
#         y = self.conv1(y)
#         y = self.conv2(y)
#         y = self.sigmoid(y)

#         return x * y
class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=8):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		# 利用1x1卷积代替全连接
		self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)


class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()

		assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1
		self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		x = torch.cat([avg_out, max_out], dim=1)
		x = self.conv1(x)
		return self.sigmoid(x)


class CSAM(nn.Module):
	def __init__(self, channels, ratio=8, kernel_size=7):
		super(CSAM, self).__init__()
		self.channelattention = ChannelAttention(channels, ratio=ratio)
		self.spatialattention = SpatialAttention(kernel_size=kernel_size)

	def forward(self, x):
		x = x * self.channelattention(x)
		x = x * self.spatialattention(x)
		return x


class CDCM(nn.Module):
	"""
	Compact Dilation Convolution based Module
	"""

	def __init__(self, in_channels, out_channels):
		super(CDCM, self).__init__()

		self.relu1 = nn.ReLU()
		self.conv1 = nn.Conv2d(in_channels, out_channels,
							   kernel_size=1, padding=0)
		self.conv2_1 = nn.Conv2d(
			out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
		self.conv2_2 = nn.Conv2d(
			out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
		self.conv2_3 = nn.Conv2d(
			out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
		self.conv2_4 = nn.Conv2d(
			out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
		nn.init.constant_(self.conv1.bias, 0)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = self.relu1(x)
		x = self.conv1(x)
		x1 = self.conv2_1(x)
		x2 = self.conv2_2(x)
		x3 = self.dropout(self.conv2_3(x))
		x4 = self.dropout(self.conv2_4(x))
		return x1 + x2 + x3 + x4


class MapReduce(nn.Module):
	"""
	Reduce feature maps into a single edge map
	"""

	def __init__(self, channels):
		super(MapReduce, self).__init__()
		self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
		self.dropout = nn.Dropout(0.2)
		nn.init.constant_(self.conv.bias, 0)

	def forward(self, x):
		return self.dropout(self.conv(x))


class PDCBlock(nn.Module):
	def __init__(self, pdc, inplane, ouplane, stride=1):
		super(PDCBlock, self).__init__()
		self.stride = stride

		self.stride = stride
		if self.stride > 1:
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
			self.shortcut = nn.Conv2d(
				inplane, ouplane, kernel_size=1, padding=0)
		self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3,
							padding=1, groups=inplane, bias=False)
		self.relu2 = nn.ReLU()
		self.conv2 = nn.Conv2d(
			inplane, ouplane, kernel_size=1, padding=0, bias=False)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		if self.stride > 1:
			x = self.pool(x)
		y = self.conv1(x)
		y = self.relu2(y)
		y = self.conv2(y)
		if self.stride > 1:
			x = self.shortcut(x)
		y = y + x
		return y


class PDCBlock_converted(nn.Module):
	"""
	CPDC, APDC can be converted to vanilla 3x3 convolution
	RPDC can be converted to vanilla 5x5 convolution
	"""

	def __init__(self, pdc, inplane, ouplane, stride=1):
		super(PDCBlock_converted, self).__init__()
		self.stride = stride

		if self.stride > 1:
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
			self.shortcut = nn.Conv2d(
				inplane, ouplane, kernel_size=1, padding=0)
		if pdc == 'rd':
			self.conv1 = nn.Conv2d(
				inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
		else:
			self.conv1 = nn.Conv2d(
				inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
		self.relu2 = nn.ReLU()
		self.conv2 = nn.Conv2d(
			inplane, ouplane, kernel_size=1, padding=0, bias=False)

	def forward(self, x):
		if self.stride > 1:
			x = self.pool(x)
		y = self.conv1(x)
		y = self.relu2(y)
		y = self.conv2(y)
		if self.stride > 1:
			x = self.shortcut(x)
		y = y + x
		return y


class PiDiNet(nn.Module):
	def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
		super(PiDiNet, self).__init__()
		self.sa = sa
		if dil is not None:
			assert isinstance(dil, int), 'dil should be an int'
		self.dil = dil
		self.dropout = nn.Dropout(0.2)
		self.fuseplanes = []

		self.inplane = inplane
		if convert:
			if pdcs[0] == 'rd':
				init_kernel_size = 5
				init_padding = 2
			else:
				init_kernel_size = 3
				init_padding = 1
			self.init_block = nn.Conv2d(1, self.inplane,
										kernel_size=init_kernel_size, padding=init_padding, bias=False)
			block_class = PDCBlock_converted
		else:
			self.init_block = Conv2d(
				pdcs[0], 1, self.inplane, kernel_size=3, padding=1)
			block_class = PDCBlock

		self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
		self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
		self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
		self.fuseplanes.append(self.inplane)  # C

		inplane = self.inplane
		self.inplane = self.inplane * 2
		self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
		self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
		self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
		self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
		self.fuseplanes.append(self.inplane)  # 2C

		inplane = self.inplane
		self.inplane = self.inplane * 2
		self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
		self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
		self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
		self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
		self.fuseplanes.append(self.inplane)  # 4C

		self.block4_1 = block_class(
			pdcs[12], self.inplane, self.inplane, stride=2)
		self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
		self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
		self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
		self.fuseplanes.append(self.inplane)  # 4C

		self.conv_reduces = nn.ModuleList()
		if self.sa and self.dil is not None:
			self.attentions = nn.ModuleList()
			self.dilations = nn.ModuleList()
			for i in range(4):
				self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
				print("self.dil:{}".format(self.dil))
				self.attentions.append(CSAM(self.dil))
				self.conv_reduces.append(MapReduce(self.dil))
		elif self.sa:
			self.attentions = nn.ModuleList()
			for i in range(4):
				self.attentions.append(CSAM(self.fuseplanes[i]))
				self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
		elif self.dil is not None:
			self.dilations = nn.ModuleList()
			for i in range(4):
				self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
				self.conv_reduces.append(MapReduce(self.dil))
		else:
			for i in range(4):
				self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

		self.classifier = nn.Conv2d(4, 1, kernel_size=1)  # has bias
		nn.init.constant_(self.classifier.weight, 0.25)
		nn.init.constant_(self.classifier.bias, 0)

		print('initialization done')

	def get_weights(self):
		conv_weights = []
		bn_weights = []
		relu_weights = []
		for pname, p in self.named_parameters():
			if 'bn' in pname:
				bn_weights.append(p)
			elif 'relu' in pname:
				relu_weights.append(p)
			else:
				conv_weights.append(p)

		return conv_weights, bn_weights, relu_weights

	def forward(self, x):
		H, W = x.size()[2:]

		x = self.init_block(x)

		x1 = self.block1_1(x)
		x1 = self.block1_2(x1)
		x1 = self.block1_3(x1)

		x2 = self.block2_1(x1)
		x2 = self.block2_2(x2)
		x2 = self.block2_3(x2)
		x2 = self.block2_4(x2)

		x3 = self.block3_1(x2)
		x3 = self.block3_2(x3)
		x3 = self.block3_3(x3)
		x3 = self.block3_4(x3)

		x4 = self.block4_1(x3)
		x4 = self.block4_2(x4)
		x4 = self.block4_3(x4)
		x4 = self.block4_4(x4)

		x_fuses = []
		if self.sa and self.dil is not None:
			for i, xi in enumerate([x1, x2, x3, x4]):
				x_fuses.append(self.attentions[i](self.dilations[i](xi)))
		elif self.sa:
			for i, xi in enumerate([x1, x2, x3, x4]):
				x_fuses.append(self.attentions[i](xi))
		elif self.dil is not None:
			for i, xi in enumerate([x1, x2, x3, x4]):
				x_fuses.append(self.dilations[i](xi))
		else:
			x_fuses = [x1, x2, x3, x4]

		e1 = self.conv_reduces[0](x_fuses[0])
		e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

		e2 = self.conv_reduces[1](x_fuses[1])
		e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

		e3 = self.conv_reduces[2](x_fuses[2])
		e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

		e4 = self.conv_reduces[3](x_fuses[3])
		e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

		outputs = [e1, e2, e3, e4]

		output = self.classifier(torch.cat(outputs, dim=1))
		# if not self.training:
		#    return torch.sigmoid(output)

		outputs.append(output)
		outputs = [torch.sigmoid(r) for r in outputs]
		return outputs


def pidinet_tiny(config, dil, sa):
	pdcs = config_model(config)
	dil = 8 if dil else None
	return PiDiNet(20, pdcs, dil=dil, sa=sa)


def pidinet_small(config, dil, sa):
	pdcs = config_model(config)
	dil = 12 if dil else None
	return PiDiNet(30, pdcs, dil=dil, sa=sa)


def pidinet(config, dil, sa):
	pdcs = config_model(config)
	# print(pdcs)
	dil = 24 if dil else None
	return PiDiNet(60, pdcs, dil=dil, sa=sa)


# convert pidinet to vanilla cnn

def pidinet_tiny_converted(args):
	pdcs = config_model_converted(args.config)
	dil = 8 if args.dil else None
	return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)


def pidinet_small_converted(args):
	pdcs = config_model_converted(args.config)
	dil = 12 if args.dil else None
	return PiDiNet(30, pdcs, dil=dil, sa=args.sa, convert=True)


class faultsDataset(torch.utils.data.Dataset):
	'''dataset for faults'''

	def __init__(self, imgs_dir, masks_dir):
		#         self.train = train
		self.images_dir = imgs_dir
		self.masks_dir = masks_dir
		self.ids = [splitext(file)[0] for file in listdir(
			imgs_dir) if not file.startswith('.')]

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, i):
		idx = self.ids[i]
		mask = np.load("{}/{}.npy".format(self.masks_dir, idx))
		img = np.load("{}/{}.npy".format(self.images_dir, idx))
		#         mask_file = glob(self.masks_dir + idx + '.npy')
		#         img_file = glob(self.images_dir + idx + '.npy')

		#         assert len(mask_file) == 1, \
		#             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
		#         assert len(img_file) == 1, \
		#             f'Either no image or multiple images found for the ID {idx}: {img_file}'
		#         mask = np.load(mask_file[0])
		#         img = np.load(img_file[0])

		assert img.size == mask.size, \
			f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

		img = np.expand_dims(img, axis=0)
		mask = np.expand_dims(mask, axis=0)
		# print("img.shape:{0},mask.shape:{1}".format(img.shape,mask.shape))
		return (img, mask)


class LossHistory():
	def __init__(self, log_dir):
		import datetime
		curr_time = datetime.datetime.now()
		time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
		self.log_dir = log_dir
		self.time_str = time_str
		self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
		global sp
		sp = self.save_path
		self.losses = []
		self.val_loss = []
		print("save_path:{}".format(self.save_path))
		os.makedirs(self.save_path)

	def append_loss(self, loss, val_loss):
		self.losses.append(loss)
		self.val_loss.append(val_loss)
		with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
			f.write(str(loss))
			f.write("\n")
		with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
			f.write(str(val_loss))
			f.write("\n")
		self.loss_plot()

	def loss_plot(self):
		iters = range(len(self.losses))

		plt.figure()
		plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
		plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
		try:
			if len(self.losses) < 25:
				num = 5
			else:
				num = 15

			plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3),
					 'green', linestyle='--', linewidth=2, label='smooth train loss')
			plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3),
					 '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
		except:
			pass

		plt.grid(True)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend(loc="upper right")

		plt.savefig(os.path.join(self.save_path,
								 "epoch_loss_" + str(self.time_str) + ".png"))

		plt.cla()
		plt.close("all")


SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
	# You can comment out this line if you are passing tensors of equal shape
	# But if you are passing output from UNet or something it will most probably
	# be with the BATCH x 1 x H x W shape

	outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

	intersection = (outputs & labels).float().sum(
		(1, 2))  # Will be zero if Truth=0 or Prediction=0
	union = (outputs | labels).float().sum(
		(1, 2))  # Will be zzero if both are 0

	# We smooth our devision to avoid 0/0
	iou = (intersection + SMOOTH) / (union + SMOOTH)
	return iou


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


bceloss = nn.BCEWithLogitsLoss()


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
				  cuda):
	total_loss = 0
	total_dice_loss = 0
	total_dice_loss_val = 0
	train_accuracies = []
	val_accuracies = []
	total_dice_loss_val = 0
	val_loss = 0
	total_acc = 0
	val_acc = 0
	model_train.train()
	print('Start Train')
	with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
		for iteration, batch in enumerate(gen):
			if iteration >= epoch_step:
				break
			images, masks = batch
			# print(type(imgs))
			with torch.no_grad():
				torch.cuda.empty_cache()
				images = Variable(images.cuda())
				# images=images.half()
				masks = Variable(masks.cuda())
				# masks=masks.half()
				if True:
					imgs = images.cuda()
					masks = masks.cuda()
	optimizer.zero_grad()
	outputs = model_train(imgs)
	if not isinstance(outputs, list):
		loss = cross_entropy_loss_RCF(outputs, masks)
	else:
		loss = 0
		for o in outputs:
			loss += cross_entropy_loss_RCF(o, masks)
		y_preds = outputs[-1]
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	predicted_mask = outputs > 0.5
	train_acc = iou_pytorch(predicted_mask.squeeze(
		1).byte(), masks.squeeze(1).byte())

	total_loss += loss.item()
	# total_dice_loss+=dice_loss1.item()
	# print(type( _dice_coeff))
	# total_dice_coeff += _dice_coeff
	# total_acc += acc.item()
	train_accuracies.append(train_acc.mean())
	pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
						# 'total_dice_loss': total_dice_loss/(iteration+1),
						'lr': get_lr(optimizer)})
	pbar.update(1)


print('Finish Train')

model_train.eval()
print('Start Validation')
with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
	for iteration, batch in enumerate(gen_val):
		if iteration >= epoch_step_val:
			break
		imgs, mask = batch
		with torch.no_grad():
			images = Variable(imgs.cuda())
			# images=images.half()
			masks = Variable(mask.cuda())
			# masks=masks.half()
			if cuda:
				imgs = images.cuda()
				masks = masks.cuda()

			outputs = model_train(imgs)

			# loss = calc_loss(outputs, masks)
			# loss = bceloss(outputs, masks)
			# dice_loss1=Dice_loss(outputs, masks)
			loss = dice_loss(outputs, masks)
			# -------------------------------#
			#   计算f_score
			# -------------------------------#
			# dice_coeff = Dice_coeff(outputs, mask)
			predicted_mask = outputs > 0.5

			train_acc_val = iou_pytorch(predicted_mask.squeeze(
				1).byte(), masks.squeeze(1).byte())
			val_loss += loss.item()
			# total_dice_loss_val+=dice_loss1.item()
			val_accuracies.append(train_acc_val.mean())

		pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
							# 'dice_loss_val': total_dice_loss_val/(iteration+1),
							'lr': get_lr(optimizer)})
		pbar.update(1)

loss_history.append_loss(total_loss / (epoch_step + 1),
						 val_loss / (epoch_step_val + 1))
print('Finish Validation')
print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
print('Total Loss: %.3f || Val Loss: %.3f ' %
	  (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
print("train_IOU:{} || val_IOU:{}".format(torch.mean(torch.stack(
	train_accuracies)), torch.mean(torch.stack(val_accuracies))))
print(sp + '/ep%03d-loss%.3f-val_loss%.3f.pth')
torch.save(model.state_dict(), sp + '/ep%03d-loss%.3f-val_loss%.3f.pth' %
		   ((epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))

# =========================================================#
# =========================================================#
#############main function########
# =========================================================#
# =========================================================#
if __name__ == "__main__":
	# -------------------------------#
	#   是否使用Cuda
	#   没有GPU可以设置成False
	# -------------------------------#
	Cuda = True
	# -------------------------------#
	#   训练自己的数据集必须要修改的
	#   自己需要的分类个数+1，如2+1
	# -------------------------------#
	num_classes = 1
	#   是否给不同种类赋予不同的损失权值，默认是平衡的。
	#   设置的话，注意设置成numpy形式的，长度和num_classes一样。
	#   如：
	#   num_classes = 3
	#   cls_weights = np.array([1, 2, 3], np.float32)
	# ---------------------------------------------------------------------#
	# ------------------------------#
	# 批量大小
	# ------------------------------#
	batch_size = 32
	# -------------------------------#
	# 学习率
	# ------------------------------#
	lr = 0.00001
	# ------------------------------#
	# 训练轮数
	# ------------------------------#
	Epoch = 50
	# ------------------------------#
	#    加载权重
	# -------------------------------#
	model_path = ""
	# ------------------------------#
	#   输入图片的大小
	# ------------------------------#
	input_shape = [96, 96]
	# ------------------------------#
	#   数据集路径
	# ------------------------------#
	pretrained = False
	data_path = "/data/max/auyu's code/dataset/processedThebe"
	# ---------------------------------------------------------------------#
	#   建议选项：
	#   种类少（几类）时，设置为True
	#   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
	#   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
	# ---------------------------------------------------------------------#

	# ---------------------------------------------------------------------#
	#   是否使用focal loss来防止正负样本不平衡
	# ---------------------------------------------------------------------#
	focal_loss = False
	# ------------------------------------------------------#
	#   用于设置是否使用多线程读取数据
	#   开启后会加快数据读取速度，但是会占用更多内存
	#   内存较小的电脑可以设置为2或者0
	# ------------------------------------------------------#
	num_workers = 8
	# ------------------------------------------------------#
	# 定义模型
	# ------------------------------
	model = pidinet(config='carv4', dil=True, sa=True)
	# model=model.half()
	model = model.cuda()
	model_train = model.train()
	if Cuda:
		model_train = torch.nn.DataParallel(model)
		cudnn.benchmark = True
		model_train = model_train.cuda()
	loss_history = LossHistory("logs_ResUnet/")
	print(sp + '/ep%03d-loss%.3f-val_loss%.3f.pth')
	print(type(sp))
	# =============================================================#
	import os

	path = "/data/max/auyu's code/dataset/processedThebe/train/seismic"
	path1 = "/data/max/auyu's code/dataset/processedThebe/val/seismic"
	count = 0
	for file in os.listdir(path):  # file 表示的是文件名
		count = count + 1
	train_lines = count
	count1 = 0
	for file in os.listdir(path1):  # file 表示的是文件名
		count1 = count1 + 1
	val_lines = count1
	# =============================================================#
	if not pretrained:
		weights_init(model)
	if model_path != '':
		# ------------------------------------------------------#
		#   权值文件请看README
		# ------------------------------------------------------#
		print('Load weights {}.'.format(model_path))
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model_dict = model.state_dict()
		pretrained_dict = torch.load(model_path, map_location=device)
		pretrained_dict = {k: v for k, v in pretrained_dict.items(
		) if np.shape(model_dict[k]) == np.shape(v)}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	if True:
		batch_size = batch_size
		lr = lr
		optimizer = optim.Adam(model_train.parameters(), lr, eps=1e-3)
		lr_scheduler = optim.lr_scheduler.StepLR(
			optimizer, step_size=5, gamma=0.7)
		faults_dataset_train = faultsDataset(imgs_dir="{}/train/seismic".format(data_path),
											 masks_dir="{}/train/annotation".format(data_path))
		faults_dataset_val = faultsDataset(imgs_dir="{}/val/seismic".format(data_path),
										   masks_dir="{}/val/annotation".format(data_path))
		gen = DataLoader(faults_dataset_train, shuffle=True, batch_size=batch_size, num_workers=num_workers,
						 pin_memory=True,
						 drop_last=True)
		gen_val = DataLoader(faults_dataset_val, shuffle=True, batch_size=batch_size, num_workers=num_workers,
							 pin_memory=True,
							 drop_last=True, )
		epoch_step = (train_lines) // batch_size
		epoch_step_val = (val_lines) // batch_size

		for epoch in range(0, Epoch):
			fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
						  Epoch, Cuda)
			lr_scheduler.step()
