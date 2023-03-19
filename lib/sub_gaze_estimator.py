
'''
	This file hold a class 'SubSimGAN' that has some basic
	functionality we can inherit from when building the
	TrainSimGAN or TestSimGAN classes. Most of it isn't 
	terribly important thats why I hide it in this sub class.

	Things such as accuracy metrics, data loaders, 
	weight loading, etc
'''

import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import argparse
import torch.autograd as autograd
from torchvision import transforms, models
from time import sleep
from torch.autograd import Variable
import random
import cv2
import torch.nn.functional as F
from models.style_transfer_model import Refiner1, Refiner2, Global_Discriminator1, Local_Discriminator1, Global_Discriminator2, Local_Discriminator2, Regressor, GazeEstimator
from models.simgan_models import SimGAN_Refiner, SimGAN_Discriminator
from models.cyclegan_models import CycleGAN_Refiner1, CycleGAN_Refiner2, CycleGAN_Discriminator1, CycleGAN_Discriminator2
from models.gaze_estimator_model import Hourglass_net, DenseNet
from data_loader.data_loader import Fake_Dataset, Real_Dataset, Eval_Real_Dataset, Eval_Fake_Dataset_with_clamp, Test_Data, Validation_Data, Fake_Rotation_Dataset
from tqdm import trange, tqdm
import numpy as np
from losses.losses import HeatmapLoss, GazeLoss, LandmarkLoss
from core.raft import RAFT
from util.gazemap_tensor import gazemap_generator
from PIL import Image
from torchvision import transforms, utils

import yaml

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)

class GazeEstimator:
	def __init__(self):
		
		torch.manual_seed(1)
	
		# initializing variables to None, used later
		self.R1 = None
		self.R1_past = None
		self.R2 = None
		self.G_D1 = None
		self.L_D1 = None
		self.G_D2 = None
		self.L_D2 = None
		self.Reg = None
		self.RAFT = None
		self.Hourglass_net = None
		self.Densenet = None

		
		self.refiner1 = None
		self.refiner2 = None
		self.global_discriminator1 = None
		self.local_discriminator1 = None
		self.global_discriminator2 = None
		self.local_discriminator2 = None
		self.regressor = None
		self.gaze_estimator = None

		self.refiner1_optimizer = None
		self.refiner2_optimizer = None
		self.global_discriminator_optimizer1 = None
		self.local_discriminator_optimizer1 = None
		self.global_discriminator_optimizer2 = None
		self.local_discriminator_optimizer2 = None
		self.regressor_optimizer = None
		self.gaze_estimator_optimizer = None
		self.Hourglass_net_optimizer = None
		self.Densenet_optimizer = None

		
		self.feature_loss = None #Usually L1 norm or content loss
		self.local_adversarial_loss = None #CrossEntropyLoss
		self.data_loader = None
		self.current_step = None

		self.synthetic_data_loader = None
		self.real_data_loader = None
		self.synthetic_data_iter = None
		self.test_real_data_loader = None
		self.real_data_iter = None
		self.weights_loaded = None
		self.test_weights_loaded = None
		self.current_step = 0

		# Set loss functions
		self.feature_loss = nn.L1Loss().cuda()
		self.mse_loss = nn.MSELoss()
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.bce_loss = nn.BCEWithLogitsLoss()

		if not config['train']['bool']:
			self.testing_done = False

	# used internally
	# checks for saved weights in the checkpoint path
	# return True if weights are loaded
	# return False if no weights are found
	print("sub simgan initialize finish.")
	def load_weights(self):
		
		print("Checking for Saved Weights")

		# If checkpoint path doesn't exist, create it
		if not os.path.isdir(config['checkpoint_path']['pathname']):
			os.mkdir(config['checkpoint_path']['pathname'])
		
		# get list of checkpoints from checkpoint path
		checkpoints = os.listdir(config['checkpoint_path']['pathname'])
		#print(checkpoints)
		# Only load weights that start with 'R_' or 'D_'

		refiner1_checkpoints = [ckpt for ckpt in checkpoints if 'R1_' == ckpt[:3]]
		refiner2_checkpoints = [ckpt for ckpt in checkpoints if 'R2_' == ckpt[:3]]
		global_discriminator1_checkpoints = [ckpt for ckpt in checkpoints if 'G_D1_' == ckpt[:5]]
		local_discriminator1_checkpoints = [ckpt for ckpt in checkpoints if 'L_D1_' == ckpt[:5]]
		global_discriminator2_checkpoints = [ckpt for ckpt in checkpoints if 'G_D2_' == ckpt[:5]]
		local_discriminator2_checkpoints = [ckpt for ckpt in checkpoints if 'L_D2_' == ckpt[:5]]
		regressor_checkpoints = [ckpt for ckpt in checkpoints if 'Reg_' == ckpt[:4]]
		densenet_checkpoints = [ckpt for ckpt in checkpoints if 'Densenet_' == ckpt[:9]]
		optimizer_status_checkpoints = [ckpt for ckpt in checkpoints if 'optimizer_status_' == ckpt[:17]]

		refiner1_checkpoints.sort(key=lambda x: int(x[3:-4]), reverse=True)
		refiner2_checkpoints.sort(key=lambda x: int(x[3:-4]), reverse=True)
		global_discriminator1_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		local_discriminator1_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		global_discriminator2_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		local_discriminator2_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		regressor_checkpoints.sort(key=lambda x: int(x[4:-4]), reverse=True)
		densenet_checkpoints.sort(key=lambda x: int(x[9:-4]), reverse=True)
		optimizer_status_checkpoints.sort(key=lambda x: int(x[17:-4]), reverse=True)

		if len(densenet_checkpoints) == 0 or not os.path.isfile(	
										os.path.join(config['checkpoint_path']['pathname'], config['pretrain_optimizer_path']['pathname'])):
			print("No Previous Weights Found. Building and Initializing new Model")
			self.current_step = 0
			return False

		print("Found Saved Weights, Loading...")		

		if config['train']['bool']:
			optimizer_status = torch.load(os.path.join(config['checkpoint_path']['pathname'], optimizer_status_checkpoints[0]))
			print(os.path.join(config['checkpoint_path']['pathname'], optimizer_status_checkpoints[0]))

			self.refiner1_optimizer.load_state_dict(optimizer_status['optR1'])
			self.refiner2_optimizer.load_state_dict(optimizer_status['optR2'])
			self.global_discriminator1_optimizer.load_state_dict(optimizer_status['optG_D_1'])
			self.local_discriminator1_optimizer.load_state_dict(optimizer_status['optL_D_1'])
			self.global_discriminator2_optimizer.load_state_dict(optimizer_status['optG_D_2'])
			self.local_discriminator2_optimizer.load_state_dict(optimizer_status['optL_D_2'])
			self.regressor_optimizer.load_state_dict(optimizer_status['optReg'])
			self.Densenet_optimizer.load_state_dict(optimizer_status['optDensenet'])
			self.current_step = optimizer_status['step']

			self.G_D1.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], global_discriminator1_checkpoints[0])), strict=False)
			self.L_D1.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], local_discriminator1_checkpoints[0])), strict=False)
			self.G_D2.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], global_discriminator2_checkpoints[0])), strict=False)
			self.L_D2.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], local_discriminator2_checkpoints[0])), strict=False)

		self.R1.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], refiner1_checkpoints[0])), strict=False)
		self.R2.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], refiner2_checkpoints[0])), strict=False)
		self.Reg.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], regressor_checkpoints[0])), strict=False)
		self.Densenet.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], densenet_checkpoints[0])),strict=False)
		
		return True

	def load_test_weights(self, cnt):	
		checkpoints = os.listdir(config['test_checkpoint_path']['pathname'])
		densenet_checkpoints = [ckpt for ckpt in checkpoints if 'Densenet_' == ckpt[:9]]
		densenet_checkpoints.sort(key=lambda x: int(x[9:-4]), reverse=True)
		self.check_points_steps = densenet_checkpoints[cnt]

		if len(os.listdir(config['test_checkpoint_path']['pathname'])) == 0:
			print("No Previous Weights Found. Building and Initializing new Model")
			self.current_step = 0
			return False

		print("Found Saved Weights, Loading...")		

		self.Densenet.load_state_dict(torch.load(os.path.join(config['test_checkpoint_path']['pathname'], densenet_checkpoints[cnt])), strict=False)
		print(os.path.join(config['test_checkpoint_path']['pathname'], densenet_checkpoints[cnt]))
		
		return True
	
	def refiner1_checkpoint_checker(self):
		checkpoints = os.listdir(config['Refiner1_checkpoint_path']['pathname'])
		refiner1_checkpoints = [ckpt for ckpt in checkpoints if 'R1_' == ckpt[:3]]
		if len(refiner1_checkpoints) != 0:
			return True
		else:
			return False
    		
	def load_R1_past_weights(self):
		checkpoints = os.listdir(config['Refiner1_checkpoint_path']['pathname'])
		refiner1_checkpoints = [ckpt for ckpt in checkpoints if 'R1_' == ckpt[:3]]
		if not len(refiner1_checkpoints) == 0:
			refiner1_checkpoints.sort(key=lambda x: int(x[3:-4]), reverse=False)
			random_num = random.randint(1,len(refiner1_checkpoints))
			self.R1_past.load_state_dict(torch.load(os.path.join(config['Refiner1_checkpoint_path']['pathname'], refiner1_checkpoints[random_num - 1])))
			print("Load Refiner1_past:" + refiner1_checkpoints[random_num - 1])
		else:
			return False 


	def build_test_network(self, cnt=None):
		in_channels = 1
		out_channels = 64
		stride = 2	
		self.Densenet = DenseNet(num_init_features=out_channels).cuda()

		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")

			device_id = config['gpu_num']['num']
			self.Densenet = torch.nn.DataParallel(self.Densenet, device_ids=device_id)

			self.Densenet = self.Densenet.cuda()

			self.test_weights_loaded = self.load_test_weights(cnt)


	def build_network(self, cnt=None):
		print("Building SimGAN Network")

		#SP-ST GAN
		self.R1 = Refiner1().cuda()
		self.R1_past = Refiner1().cuda()
		self.R2 = Refiner2().cuda()
		self.G_D1 = Global_Discriminator1().cuda()
		self.L_D1 = Local_Discriminator1().cuda()
		self.G_D2 = Global_Discriminator2().cuda()
		self.L_D2 = Local_Discriminator2().cuda()

		#SimGAN
		# self.R1 = SimGAN_Refiner().cuda()
		# self.R1_past = SimGAN_Refiner().cuda()
		# self.L_D1 = SimGAN_Discriminator().cuda()

		#CycleGAN
		# self.R1 = CycleGAN_Refiner1().cuda()
		# self.R1_past = CycleGAN_Refiner1().cuda()
		# self.R2 = CycleGAN_Refiner2().cuda()
		# self.L_D1 = CycleGAN_Discriminator1().cuda()
		# self.L_D2 = CycleGAN_Discriminator2().cuda()		

		#Landmark Detector & Gaze Estimator
		in_channels = 1
		out_channels = 64
		stride = 2
		self.Reg = Regressor(3, 32, 34).cuda()
		self.Densenet = DenseNet(num_init_features=out_channels).cuda()

		"""Build Raft_Network"""
		print("Load & Build RAFT...")
		parser = argparse.ArgumentParser()
		parser.add_argument('--model', help="restore checkpoint",default='models/raft-small.pth')
		parser.add_argument('--path', help="dataset for evaluation")
		parser.add_argument('--small', action='store_true', help='use small model',default='--small')
		parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
		parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
		args = parser.parse_args()
		print(args)
		self.RAFT = torch.nn.DataParallel(RAFT(args), device_ids=config['gpu_num']['num'])
		self.RAFT = RAFT(args)

		"""Load Pretrained Model"""
		model_dir = '../raft_pretrained_model'
		trained_model = os.listdir(model_dir)
		print('Use ',trained_model[0], ' for RAFT')
		# self.RAFT.load_state_dict(torch.load(os.path.join(model_dir, 'raft-small.pth')), strict=False) #None Pre-train RAFT model
		self.RAFT.load_state_dict(torch.load(os.path.join(model_dir, '1000000_raft_grayscale_finetune.pth')), strict=False)
		
		self.RAFT = self.RAFT.cuda()
		self.RAFT.eval()
		for param in self.RAFT.parameters():
				param.requires_grad = False
		
		print("Load Complete RAFT Weights...")
		
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")

			device_id = config['gpu_num']['num']
			self.R1 = torch.nn.DataParallel(self.R1, device_ids=device_id)
			self.R1_past = torch.nn.DataParallel(self.R1_past, device_ids = device_id)
			self.G_D1 = torch.nn.DataParallel(self.G_D1, device_ids=device_id)
			self.L_D1 = torch.nn.DataParallel(self.L_D1, device_ids=device_id)
			self.R2 = torch.nn.DataParallel(self.R2, device_ids=device_id)
			self.G_D2 = torch.nn.DataParallel(self.G_D2, device_ids=device_id)
			self.L_D2 = torch.nn.DataParallel(self.L_D2, device_ids=device_id)
			self.Reg = torch.nn.DataParallel(self.Reg, device_ids=device_id)
			self.Densenet = torch.nn.DataParallel(self.Densenet, device_ids=device_id)

			self.R1 = self.R1.cuda()
			self.R1_past = self.R1_past.cuda()
			self.R2 = self.R2.cuda()

			self.G_D1 = self.G_D1.cuda()
			self.L_D1 = self.L_D1.cuda()
			self.G_D2 = self.G_D2.cuda()
			self.L_D2 = self.L_D2.cuda()
			self.Reg = self.Reg.cuda()
			self.Densenet = self.Densenet.cuda()



		
		if config['train']['bool']:
			# Set optimizers
			self.refiner1_optimizer = torch.optim.Adam(self.R1.parameters(), lr=config['refiner_lr']['param'], betas=(0.5, 0.999))
			self.refiner2_optimizer = torch.optim.Adam(self.R2.parameters(), lr=config['refiner_lr']['param'], betas=(0.5, 0.999))

			self.global_discriminator1_optimizer = torch.optim.Adam(self.G_D1.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))
			self.local_discriminator1_optimizer = torch.optim.Adam(self.L_D1.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))

			self.global_discriminator2_optimizer = torch.optim.Adam(self.G_D2.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))
			self.local_discriminator2_optimizer = torch.optim.Adam(self.L_D2.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))
			
			self.regressor_optimizer = torch.optim.Adam(self.Reg.parameters(), lr=config['regressor_lr']['param'])

			self.Densenet_optimizer = torch.optim.Adam(self.Densenet.parameters(), lr = config['densenet_net_lr']['param'])

		if cnt == None:
			self.weights_loaded = self.load_weights()
		else:
			self.test_weights_loaded = self.load_test_weights(cnt)

		print('Done building')
	
	def nan_checker(self, data, name):
		nan_check = torch.isnan(data)
		if True in nan_check:
			print("Nan in" +  name)
			print(name + ":" + data)
			data_name = name
			assert True not in torch.isnan(data), 'Nan in {[0]}'.format(data_name)

	def calc_landmark_diff(self,  pred_landmarks,  gt_landmarks):
		diff_landmarks_dict = {}
		diff_landmarks = gt_landmarks - pred_landmarks
		diff_landmarks_sum = torch.sum(diff_landmarks, dim=2)
		diff_landmarks_sum = torch.abs(diff_landmarks_sum)
		diff_landmarks_mean = torch.mean(diff_landmarks_sum, dim=0)
		for i in range(34):
			key_name = 'ldmk_' + str(i)
			diff_landmarks_dict[key_name] = diff_landmarks_mean[i]

		return diff_landmarks_dict

	'''Gradient Penalty for WGAN Loss'''
	def global_compute_gradient_penalty(self, D, real_samples, fake_samples):
		"""Calculates the gradient penalty loss for WGAN GP"""
		Tensor = torch.cuda.FloatTensor
		alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
		interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
		d_interpolates = D(interpolates)
		fake = Variable(Tensor(real_samples.shape[0], 384).fill_(1.0), requires_grad=False).cuda()
		gradients = autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		global_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return global_gradient_penalty

	def local_compute_gradient_penalty(self, D, real_samples, fake_samples):
		"""Calculates the gradient penalty loss for WGAN GP"""
		Tensor = torch.cuda.FloatTensor
		alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
		# Get random interpolation between real and fake samples
		interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
		d_interpolates = D(interpolates)
		d_interpolates = torch.reshape(d_interpolates,(d_interpolates.shape[0],-1))
		# fake = Variable(Tensor(real_samples.shape[0], 17280).fill_(1.0), requires_grad=False).cuda()
		fake = Variable(Tensor(real_samples.shape[0], d_interpolates.shape[1]).fill_(1.0), requires_grad=False).cuda()
		# Get gradient w.r.t. interpolates
		gradients = autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		local_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return local_gradient_penalty	

	# iterator for the data loader..
	# not really important
	def loop_iter(self, dataloader):
		while True:
			for data in iter(dataloader):
				yield data
			
			if not config['train']['bool']:
				print('Finished one epoch, done testing')
				self.testing_done = True

	def edge_detection(self,gray):
		dev = gray.device
		k = config['edge_kernel_size']['size']
		edge_kernel = kernel = np.array([[k, k, k], [k, -k*8, k], [k, k, k]], np.float32)  # convolution filter  
		gaussian_kernel = np.array([[0.077847, 0.123317, 0.077847], [0.123317, 0.195346, 0.123317], [0.077847, 0.123317, 0.077847]], np.float32)
		sharpen_kernel = np.array([[-2, -2, -2], [-2, 31, -2], [-2, -2, -2]], np.float32) / 8.0
		edge_k = torch.as_tensor(edge_kernel.reshape(1, 1, 3, 3)).to(dev)
		gaussian_k =  torch.as_tensor(gaussian_kernel.reshape(1, 1, 3, 3)).to(dev)
		sharpen_k =  torch.as_tensor(sharpen_kernel.reshape(1, 1, 3, 3)).to(dev)
		edge_image = F.conv2d(gray, sharpen_k, padding=1)
		edge_image = F.conv2d(edge_image, edge_k, padding=1)
		edge_image = F.conv2d(edge_image,gaussian_k, padding=1)
		return edge_image

	def get_data_loaders(self):
		regressor_batch_size      = config['regressor_batch_size']['size']
		batchsize                 = config['batch_size']['size']
		validation_batch_size     = config['eval_batch_size']['size']
		test_batch_size           = config['test_batch_size']['size']
		num                       = config['num_worker']['num']

		synthetic_data                = Fake_Dataset()
		synthetic_rotation_data       = Fake_Rotation_Dataset()
		eval_synthetic_data           = Eval_Fake_Dataset_with_clamp()
		# test_data                     = Test_Data()
		real_data 				 	  = Real_Dataset()
		real_eval_data				  = Eval_Real_Dataset()
		mpiigaze_validation_data      = Validation_Data('mpiigaze_validation')
		mpiigaze_test_data            = Validation_Data('mpiigaze_test')
		ut_validation_data            = Validation_Data('ut_multiview_validation')
		ut_test_data                  = Validation_Data('ut_multiview_test')
		eth_xgaze_validation_data     = Validation_Data('eth_xgaze_validation_left')
		eth_xgaze_test_data           = Validation_Data('eth_xgaze_test_left')
		columbia_all_data             = Validation_Data('columbia_all')
		columbia_validation_data      = Validation_Data('columbia_validation')
		columbia_test_data            = Validation_Data('columbia_test')
		rt_gene_all_data              = Validation_Data('rt_gene_all')

		
		self.synthetic_data_loader           = Data.DataLoader(synthetic_data, batch_size=batchsize, shuffle=True, pin_memory=True, drop_last=True, num_workers=num, worker_init_fn=lambda x: np.random.seed())
		# self.synthetic_regressor_data_loader = Data.DataLoader(synthetic_rot_data, batch_size=regressor_batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num, worker_init_fn=lambda x: np.random.seed())
		self.synthetic_regressor_data_loader = Data.DataLoader(synthetic_rotation_data, batch_size=regressor_batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num, worker_init_fn=lambda x: np.random.seed())
		self.eval_data_loader                = Data.DataLoader(eval_synthetic_data, batch_size=validation_batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=num)
		self.real_data_loader 	 	         = Data.DataLoader(real_data, batch_size=batchsize, shuffle=True, pin_memory=True, drop_last=True, num_workers=num)
		self.real_eval_data_loader           = Data.DataLoader(real_eval_data, batch_size=175, shuffle=True, pin_memory=True, drop_last=True, num_workers=num)
		self.synthe_eval_data_loader         = Data.DataLoader(synthetic_data, batch_size=validation_batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num)
		# self.test_data_loader                = Data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=num)
		self.mpii_validation_data_loader     = Data.DataLoader(mpiigaze_validation_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.mpii_test_data_loader           = Data.DataLoader(mpiigaze_test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.ut_validation_data_loader       = Data.DataLoader(ut_validation_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.ut_test_data_loader             = Data.DataLoader(ut_test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.eth_validation_data_loader      = Data.DataLoader(eth_xgaze_validation_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.eth_test_data_loader            = Data.DataLoader(eth_xgaze_test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.columbia_all_data_loader        = Data.DataLoader(columbia_all_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.columbia_test_data_loader       = Data.DataLoader(columbia_test_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.columbia_validation_data_loader = Data.DataLoader(columbia_validation_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)
		self.rt_gene_all_data_loader         = Data.DataLoader(rt_gene_all_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num)

		self.synthetic_data_iter           = self.loop_iter(self.synthetic_data_loader)
		self.synthetic_regressor_data_iter = self.loop_iter(self.synthetic_regressor_data_loader)
		self.synthe_eval_data_iter         = self.loop_iter(self.synthe_eval_data_loader)
		self.real_data_iter                = self.loop_iter(self.real_data_loader)
		self.real_eval_data_iter           = self.loop_iter(self.real_eval_data_loader)
		self.eval_data_iter                = self.loop_iter(self.eval_data_loader)
		# self.test_data_iter                = self.loop_iter(self.test_data_loader)
		self.mpii_validation_data_iter     = self.loop_iter(self.mpii_validation_data_loader)
		self.mpii_test_data_iter           = self.loop_iter(self.mpii_test_data_loader)
		self.ut_validation_data_iter       = self.loop_iter(self.ut_validation_data_loader)
		self.ut_test_data_iter             = self.loop_iter(self.ut_test_data_loader)
		self.eth_validation_data_iter      = self.loop_iter(self.eth_validation_data_loader)
		self.eth_test_data_iter            = self.loop_iter(self.eth_test_data_loader)
		self.columbia_all_data_iter        = self.loop_iter(self.columbia_all_data_loader)
		self.columbia_test_iter     	   = self.loop_iter(self.columbia_test_data_loader)
		self.columbia_validation_iter      = self.loop_iter(self.columbia_validation_data_loader)
		self.rt_gene_all_data_iter         = self.loop_iter(self.rt_gene_all_data_loader)
	
	