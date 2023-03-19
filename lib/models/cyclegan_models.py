'''
	This file hold the network architecture for SimGAN including the 
	Refiner and Discriminator
'''

#from torch import nn
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import argparse
import yaml

from models.layers import Conv, Hourglass, Pool, Residual


with open('config.yaml', 'r') as yml:
    config = yaml.load(yml,Loader=yaml.Loader)

nstack = config['nstack']['param']
nfeatures = config['nfeatures']['param']
nlandmarks = config['nlandmarks']['param']

class ResidualBlock(nn.Module):
	"""Some Information about ResidualBlock"""
	def __init__(self):
		super(ResidualBlock, self).__init__()
		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(256,256,kernel_size=3,stride=1),
			nn.InstanceNorm2d(256),
			nn.ReLU(inplace=True),
			
			nn.ReflectionPad2d(1),
			nn.Conv2d(256,256,kernel_size=3,stride=1),
			nn.InstanceNorm2d(256)
		)

	def forward(self, x):
		x = x + self.block(x)
		return x

class CycleGAN_Refiner1(nn.Module):

	def __init__(self, img_channel= 1, res_block=6):
		super(Refiner1, self).__init__()
		
		''' Image size is [1, 35, 55] '''
		self.encode_block = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(1,64,kernel_size=7,stride=1),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True),

			nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1, bias=True),
			nn.InstanceNorm2d(128),
			nn.ReLU(inplace=True),

			nn.Conv2d(128,256,kernel_size=3,stride=2, padding=1, bias=True),
			nn.InstanceNorm2d(256),
			nn.ReLU(inplace=True),			
		)
		res_blocks = [ResidualBlock() for _ in range(res_block)]
		self.res_block = nn.Sequential(
			*res_blocks
		)
		self.decode_block = nn.Sequential(
			nn.ConvTranspose2d(256,128,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
			nn.InstanceNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(128,64,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True),

			# nn.ReflectionPad2d(3),
			nn.Conv2d(64,img_channel,kernel_size=2,stride=2),
			nn.Tanh()
		)

	def forward(self, x):
		x = self.encode_block(x)
		x = self.res_block(x)
		x = self.decode_block(x)
		return x


class CycleGAN_Refiner2(nn.Module):

	def __init__(self, img_channel= 1, res_block=6):
		super(Refiner2, self).__init__()
		
		''' Image size is [1, 35, 55] '''
		self.encode_block = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(1,64,kernel_size=7,stride=1),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True),

			nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1, bias=True),
			nn.InstanceNorm2d(128),
			nn.ReLU(inplace=True),

			nn.Conv2d(128,256,kernel_size=3,stride=2, padding=1, bias=True),
			nn.InstanceNorm2d(256),
			nn.ReLU(inplace=True),			
		)
		res_blocks = [ResidualBlock() for _ in range(res_block)]
		self.res_block = nn.Sequential(
			*res_blocks
		)
		self.decode_block = nn.Sequential(
			nn.ConvTranspose2d(256,128,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
			nn.InstanceNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(128,64,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True),

			# nn.ReflectionPad2d(3),
			nn.Conv2d(64,img_channel,kernel_size=2,stride=2),
			nn.Tanh()
		)
	def forward(self, x):
		x = self.encode_block(x)
		x = self.res_block(x)
		x = self.decode_block(x)
		return x

class CycleGAN_Discriminator1(nn.Module):

	def __init__(self):
		super(Local_Discriminator1, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.block = nn.Sequential(
			nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=True),
			nn.InstanceNorm2d(128),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=True),
			nn.InstanceNorm2d(256),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1,bias=True),
			nn.InstanceNorm2d(512),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)
		)
		

	def forward(self, x):
		x = self.block(x)
		return x

class CycleGAN_Discriminator2(nn.Module):

	def __init__(self):
		super(Local_Discriminator2, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.block = nn.Sequential(
			nn.Conv2d(1,64,kernel_size=4,stride=2,padding=1),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=True),
			nn.InstanceNorm2d(128),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=True),
			nn.InstanceNorm2d(256),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1,bias=True),
			nn.InstanceNorm2d(512),
			nn.LeakyReLU(0.2,inplace=True),

			nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)
		)
		

	def forward(self, x):
		x = self.block(x)
		return x


