'''
	DataLoader over rides the torch.utils.data.dataset.Dataset class
	This should be changed dependent on the format of your data sets.
'''

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageStat
import os
import sys
import json 
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from tqdm import tqdm
from util.preprocess_unityeyes import preprocess_unityeyes_image
import pandas as pd
import torch
import random

seed_num = 2
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
random.seed(seed_num)

with open('config.yaml', 'r') as yml:
    config = yaml.load(yml)
fake_dir = config['synthetic_image_directory']['dirname']
fake_num = len(os.listdir(fake_dir))
eval_fake_dir = config['eval_synthetic_image_directory']['dirname']
eval_fake_num = len(os.listdir(eval_fake_dir))

def convert_gray(self):
		self.transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1)
			]					    
		)


name_num = 1
sum_mean = 0 
sum_variance = 0

name_num = 1
sum_mean = 0 
sum_variance = 0

while name_num <= fake_num:
	img = Image.open(os.path.join(fake_dir, "{}.png".format(name_num)))
	gray_img = np.array(img.convert('L'))
	gray_img = gray_img / 255
	mean = np.mean(gray_img)
	var = np.var(gray_img)
	sum_mean = sum_mean + mean
	sum_variance = sum_variance + var
	name_num = name_num + 1    

fake_mean = sum_mean / fake_num
fake_variance = sum_variance / fake_num
# print('fake_num:' + str(fake_num))
# print('fake_mean:' + str(fake_mean))
# print('fake_variance:' + str(fake_variance))

name_num = 1
sum_mean = 0 
sum_variance = 0

name_num = 1
eval_sum_mean = 0 
eval_sum_variance = 0

while name_num <= eval_fake_num:
	img = Image.open(os.path.join(eval_fake_dir, "{}.png".format(name_num)))
	gray_img = np.array(img.convert('L'))
	gray_img = gray_img / 255
	mean = np.mean(gray_img)
	var = np.var(gray_img)
	
	eval_sum_mean = eval_sum_mean + mean
	eval_sum_variance = eval_sum_variance + var
	name_num = name_num + 1    

eval_fake_mean = eval_sum_mean / eval_fake_num
eval_fake_variance = eval_sum_variance / eval_fake_num
print('eval_fake_num:' + str(eval_fake_num))
print('eval_fake_mean:' + str(eval_fake_mean))
print('eval_fake_variance:' + str(eval_fake_variance))

class Real_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['real_image_directory']['dirname']
		self.dir_name = config['data_directory']['dirname']
		
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
			 #transforms.Resize(size=(48, 80), interpolation=2),
			 transforms.ToTensor()
			 #transforms.Normalize((real_mean,), (real_variance,))	
			]					    
		)
		#self.data_len = len(self.dir_name + "real_yzk_reg")
		self.data_len = len(self.imageFiles)
		print("real data len = " + str(self.data_len))

	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		image_files = Image.open(self.img_path + image_file)
		r_mean, r_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		
		return image_as_tensor

class Eval_Real_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['real_eval_directory']['dirname']
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
			#transforms.Resize(size=(48, 80), interpolation=2),
			transforms.ToTensor()
			#transforms.Normalize((real_mean,), (real_variance,))	
			]					    
		)
		#self.data_len = len(self.dir_name + "real_yzk_reg")
		self.data_len = len(self.imageFiles)
		print("real data len = " + str(self.data_len))

	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		image_files = Image.open(self.img_path + image_file)
		r_mean, r_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		
		return image_as_tensor

class Fake_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['synthetic_image_directory']['dirname']
		self.json_dir_name = config['synthetic_json_directory']['dirname']
		self.img_path = self.img_dir_name
		self.json_path = self.json_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])
		self.jsonFiles = sorted([json for json in os.listdir(self.json_path)])

		self.img_data_len = len(self.imageFiles)
		self.json_data_len = len(self.jsonFiles)
		self.data_len = len(self.img_dir_name)

		
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(54, 90), interpolation=2),
				#transforms.Resize(size=(48, 80), interpolation=2),
				#transforms.Normalize(fake_mean, fake_variance),
				transforms.ToTensor()
				#transforms.Normalize((f_mean,), (f_variance,)),
				#transforms.Normalize(((-1 * fake_mean / fake_variance),), ((1.0 / fake_variance),))
			]							    
		)
		self.data_len = self.img_data_len
		print('initialize finish')		
	
	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		json_file = self.jsonFiles[index]
		
		image_files = Image.open(self.img_path + image_file)
		image_files = image_files.convert("L")
		with open(self.json_path + json_file) as f:
			json_data = json.load(f)
		heatmaps, landmarks, gaze = preprocess_unityeyes_image(image_files, json_data)
		f_mean, f_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		image_as_tensor = transforms.Normalize((f_mean,), (f_variance,))(image_as_tensor)
		image_as_tensor = transforms.Normalize((-1 * fake_mean / fake_variance), (1.0 / fake_variance))(image_as_tensor)
		image_as_tensor = torch.clamp(image_as_tensor, 0, 1)

		return image_as_tensor, heatmaps, landmarks, gaze


	def __len__(self):	
		return self.data_len

class Fake_Rotation_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['synthetic_rot_image_directory']['dirname']
		self.json_dir_name = config['synthetic_rot_json_directory']['dirname']
		self.img_path = self.img_dir_name
		self.json_path = self.json_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])
		self.jsonFiles = sorted([json for json in os.listdir(self.json_path)])

		self.img_data_len = len(self.imageFiles)
		self.json_data_len = len(self.jsonFiles)
		self.data_len = len(self.img_dir_name)

		
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(54, 90), interpolation=2),
				#transforms.Resize(size=(48, 80), interpolation=2),
				#transforms.Normalize(fake_mean, fake_variance),
				transforms.ToTensor()
				#transforms.Normalize((f_mean,), (f_variance,)),
				#transforms.Normalize(((-1 * fake_mean / fake_variance),), ((1.0 / fake_variance),))
			]							    
		)
		self.data_len = self.img_data_len
		print('initialize finish')		
	
	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		json_file = self.jsonFiles[index]
		
		image_files = Image.open(self.img_path + image_file)
		image_files = image_files.convert("L")
		with open(self.json_path + json_file) as f:
			json_data = json.load(f)
		heatmaps, landmarks, gaze = preprocess_unityeyes_image(image_files, json_data)
		f_mean, f_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		image_as_tensor = transforms.Normalize((f_mean,), (f_variance,))(image_as_tensor)
		image_as_tensor = transforms.Normalize((-1 * fake_mean / fake_variance), (1.0 / fake_variance))(image_as_tensor)
		image_as_tensor = torch.clamp(image_as_tensor, 0, 1)

		return image_as_tensor, heatmaps, landmarks, gaze


	def __len__(self):	
		return self.data_len

class Eval_Fake_Dataset_with_clamp(Dataset):
	def __init__(self):
		self.img_dir_name = config['eval_synthetic_image_directory']['dirname']
		self.json_dir_name = config['eval_synthetic_json_directory']['dirname']
		self.img_path = self.img_dir_name
		self.json_path = self.json_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])
		self.jsonFiles = sorted([json for json in os.listdir(self.json_path)])
		
		self.img_data_len = len(self.imageFiles)
		#print("img_data_len = " + str(self.img_data_len))
		self.json_data_len = len(self.jsonFiles)
		#print("json_data_len = " + str(self.json_data_len))
		self.data_len = len(self.img_dir_name)

		#f_mean, f_var = self.calc_mean(self.imageFiles)

		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(54, 90), interpolation=2),
				#transforms.Resize(size=(48, 80), interpolation=2),
				#transforms.Normalize(fake_mean, fake_variance),
				transforms.ToTensor(),
				#transforms.Normalize((f_mean,), (f_variance,)),
				#transforms.Normalize(((-1 * fake_mean / fake_variance),), ((1.0 / fake_variance),))
			]							    
		)
		self.data_len = self.img_data_len
		print('initialize finish')		
	
	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		json_file = self.jsonFiles[index]
		image_files = Image.open(self.img_path + image_file)
		with open(self.json_path + json_file) as f:
			json_data = json.load(f)
		heatmaps, landmarks, gaze = preprocess_unityeyes_image(image_files, json_data)
		f_mean, f_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		image_as_tensor = transforms.Normalize((f_mean,), (f_variance,))(image_as_tensor)
		image_as_tensor = transforms.Normalize((-1 * eval_fake_mean / eval_fake_variance), (1.0 / eval_fake_variance))(image_as_tensor)
		image_as_tensor = torch.clamp(image_as_tensor,0, 1)

		return image_as_tensor, heatmaps, landmarks, gaze
		
	def __len__(self):	
		return self.data_len

class Test_Data(Dataset):
	def __init__(self):
		self.img_dir_name = config['real_image_test_directory']['dirname']
		self.dir_name = config['data_directory']['dirname']
		self.test_img_dir = config['real_image_test_directory']['dirname']
		self.img_dir = config['real_image_test_directory']['dirname']
		
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.images = []

		for image in self.imageFiles:
			image = image[:-4]
			self.images.append(image)

		self.images = sorted(self.images, key = int)
		# print(self.images)

		self.gaze_list = self.read_gt_data()

		if len(self.imageFiles) != len(self.gaze_list):
			sys.exit('data_loader TEST_Data Please confilm real test data folder name!')

		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(48, 80), interpolation=2),
				transforms.ToTensor()
				#transforms.Normalize((real_mean,), (real_variance,))	
			]					    
		)
		#self.data_len = len(self.dir_name + "real_yzk_reg")
		self.data_len = len(self.imageFiles)
		print("real data len = " + str(self.data_len))
		
	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	'''Extruct test gaze direction from CSV'''
	def read_gt_data(self):
		self.img_dir = config['real_image_test_directory']['dirname']
		self.gt_data_path = config['real_image_test_csv']['csvname']

		with open(self.gt_data_path) as self.gt_f:
			self.input_gt_datas = self.gt_f.readlines()

			self.gt_data_list = []

			for self.gt_line_datas in self.input_gt_datas:
				self.gt_datas = self.gt_line_datas.rstrip()
				self.gt_data = self.gt_datas.split(",")
				self.gt_data_list.append(self.gt_data)

			return self.gt_data_list

	def __getitem__(self, index):
		# image_file = self.imageFiles[index]
		if self.img_dir == '../data/ut_multiview_left_img/' or self.img_dir == '../data/ut_multiview_validation_data/':
			image_file = str(self.images[index]) + '.bmp'
		else:
			image_file = str(self.images[index]) + '.png'
		image_files = Image.open(self.img_path + image_file)
		image_as_tensor = self.data_transform(image_files)
		gaze = self.gaze_list[index]

		# import pdb;pdb.set_trace()
				
		return image_as_tensor, gaze, index

class Validation_Data(Dataset):
	def __init__(self, validation_data_name):
		# self.img_dir_name = config['real_image_validation_directory']['dirname']
		self.img_dir_name = '../data/' + validation_data_name + '/'
		self.dir_name = config['data_directory']['dirname']
		
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.images = []

		for image in self.imageFiles:
			image = image[:-4]
			self.images.append(image)

		self.images = sorted(self.images, key = int)
		# print(self.images)

		self.gaze_list = self.read_gt_data(validation_data_name)

		if len(self.imageFiles) != len(self.gaze_list):
			sys.exit('data_loader Validation_Data Please confilm real test data folder name!')
			
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(48, 80), interpolation=2),
				transforms.ToTensor()
				#transforms.Normalize((real_mean,), (real_variance,))	
			]					    
		)
		#self.data_len = len(self.dir_name + "real_yzk_reg")
		self.data_len = len(self.imageFiles)
		print("real data len = " + str(self.data_len))
		
	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	'''Extruct test gaze direction from CSV'''
	def read_gt_data(self, validation_data_name):
		self.gt_data_path = '../data/' + validation_data_name + '.csv'
		with open(self.gt_data_path) as self.gt_f:
			self.input_gt_datas = self.gt_f.readlines()

			self.gt_data_list = []

			for self.gt_line_datas in self.input_gt_datas:
				self.gt_datas = self.gt_line_datas.rstrip()
				self.gt_data = self.gt_datas.split(",")
				self.gt_data_list.append(self.gt_data)

			return self.gt_data_list

	def __getitem__(self, index):
		# image_file = self.imageFiles[index]
		if self.img_dir_name[:20] == '../data/ut_multiview':
			image_file = str(self.images[index]) + '.bmp'
		else:
			image_file = str(self.images[index]) + '.png'
		image_files = Image.open(self.img_path + image_file)
		image_as_tensor = self.data_transform(image_files)
		gaze = self.gaze_list[index]

		# import pdb;pdb.set_trace()
				
		return image_as_tensor, gaze, index

class Personal_Train_Data(Dataset):
	def __init__(self, validation_data_name):
		# self.img_dir_name = config['real_image_validation_directory']['dirname']
		self.img_dir_name = '../data/' + validation_data_name + '/'
		self.dir_name = config['data_directory']['dirname']
		
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.images = []

		for image in self.imageFiles:
			image = image[:-4]
			self.images.append(image)

		self.images = sorted(self.images, key = int)
		# print(self.images)

		self.gaze_list = self.read_gt_data(validation_data_name)

		if len(self.imageFiles) != len(self.gaze_list):
			sys.exit('data_loader Personal_Train_Data Please confilm real test data folder name!')
			
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(48, 80), interpolation=2),
				transforms.ToTensor()
				#transforms.Normalize((real_mean,), (real_variance,))	
			]					    
		)
		#self.data_len = len(self.dir_name + "real_yzk_reg")
		self.data_len = len(self.imageFiles)
		print("real data len = " + str(self.data_len))
		
	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	'''Extruct test gaze direction from CSV'''
	def read_gt_data(self, validation_data_name):
		self.gt_data_path = '../data/mpiigaze_' + validation_data_name[-3:] + '_train.csv'
		with open(self.gt_data_path) as self.gt_f:
			self.input_gt_datas = self.gt_f.readlines()

			self.gt_data_list = []

			for self.gt_line_datas in self.input_gt_datas:
				self.gt_datas = self.gt_line_datas.rstrip()
				self.gt_data = self.gt_datas.split(",")
				self.gt_data_list.append(self.gt_data)

			return self.gt_data_list

	def __getitem__(self, index):
		# image_file = self.imageFiles[index]
		if self.img_dir_name[:20] == '../data/ut_multiview':
			image_file = str(self.images[index]) + '.bmp'
		else:
			image_file = str(self.images[index]) + '.png'
		image_files = Image.open(self.img_path + image_file)
		image_as_tensor = self.data_transform(image_files)
		gaze = self.gaze_list[index]

		# import pdb;pdb.set_trace()
				
		return image_as_tensor, gaze, index

class Personal_Test_Data(Dataset):
	def __init__(self, validation_data_name):
		# self.img_dir_name = config['real_image_validation_directory']['dirname']
		self.img_dir_name = '../data/' + validation_data_name + '/'
		self.dir_name = config['data_directory']['dirname']
		
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.images = []

		for image in self.imageFiles:
			image = image[:-4]
			self.images.append(image)

		self.images = sorted(self.images, key = int)
		# print(self.images)

		self.gaze_list = self.read_gt_data(validation_data_name)

		if len(self.imageFiles) != len(self.gaze_list):
			sys.exit('data_loader Personal_Test_Data Please confilm real test data folder name!')
			
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				#transforms.Resize(size=(48, 80), interpolation=2),
				transforms.ToTensor()
				#transforms.Normalize((real_mean,), (real_variance,))	
			]					    
		)
		#self.data_len = len(self.dir_name + "real_yzk_reg")
		self.data_len = len(self.imageFiles)
		print("real data len = " + str(self.data_len))
		
	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	'''Extruct test gaze direction from CSV'''
	def read_gt_data(self, validation_data_name):
		self.gt_data_path = '../data/mpiigaze_' + validation_data_name[-4:-1] + '_test.csv'
		with open(self.gt_data_path) as self.gt_f:
			self.input_gt_datas = self.gt_f.readlines()

			self.gt_data_list = []

			for self.gt_line_datas in self.input_gt_datas:
				self.gt_datas = self.gt_line_datas.rstrip()
				self.gt_data = self.gt_datas.split(",")
				self.gt_data_list.append(self.gt_data)

			return self.gt_data_list

	def __getitem__(self, index):
		# image_file = self.imageFiles[index]
		if self.img_dir_name[:20] == '../data/ut_multiview':
			image_file = str(self.images[index]) + '.bmp'
		else:
			image_file = str(self.images[index]) + '.png'
		image_files = Image.open(self.img_path + image_file)
		image_as_tensor = self.data_transform(image_files)
		gaze = self.gaze_list[index]

		# import pdb;pdb.set_trace()
				
		return image_as_tensor, gaze, index