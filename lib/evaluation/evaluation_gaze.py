import torch
import torch.nn as nn
from torchvision import transforms, utils
import numpy as np
import sys
import os
from tqdm import tqdm
import yaml
import cv2
from PIL import Image
from torchvision.utils import save_image

with open('config.yaml', 'r') as yml:
    config = yaml.load(yml,Loader=yaml.Loader)

class EvaluationGazeEstimator:
	def __init__(self, densenet, current_step):
		self.Densenet = densenet
		self.current_step = current_step
	
	def draw_gaze(self, img, gaze_gt , gaze_pred):
		iris_center_x = img.shape[1] /2
		iris_center_y = img.shape[0] /2
		# import pdb;pdb.set_trace()

		gaze_pred = torch.squeeze(gaze_pred)
		gaze_gt_x = gaze_gt[0][0]
		gaze_gt_y = gaze_gt[0][1]
		gaze_pred_x = gaze_pred[0][0]
		gaze_pred_y = gaze_pred[0][1]
		
		arrow_gt_x = iris_center_x + (30 * gaze_gt_x)
		arrow_gt_y = iris_center_y + (30 * gaze_gt_y)
		arrow_pred_x = iris_center_x + (30 * gaze_pred_x)
		arrow_pred_y = iris_center_y + (30 * gaze_pred_y)

		img = cv2.arrowedLine(img, (int(iris_center_x), int(iris_center_y)), (int(arrow_gt_x), int(arrow_gt_y)), (255, 0, 0), thickness=1)
		img = cv2.arrowedLine(img, (int(iris_center_x), int(iris_center_y)), (int(arrow_pred_x), int(arrow_pred_y)), (0, 255, 0), thickness=1)

		return img

	def gaze_diff(self, gt_gaze, gaze):
		gt_gaze = gt_gaze.cuda()
		gaze = gaze.cuda()
		cos = nn.CosineSimilarity()
		'''Normalize gaze direction'''
		# normalized_gaze = torch.nn.functional.normalize(gaze, dim=2)
		gaze = torch.squeeze(gaze)
		cos_sim = cos(gt_gaze, gaze)
		cos_sim = cos_sim.to('cpu').detach().numpy().copy()
		cos_sim_rad = np.arccos(cos_sim)
		diff_gaze = cos_sim_rad * (180 / np.pi)

		return diff_gaze

	def evaluation_gaze_estimator(self, validation_data_name, data_iter):
		"""Initialize"""
		img_num = 1
		diff_gaze_array = np.empty(0)
		result_list = []
		diff_gaze_list = []
		img_tensor_list = []

		self.img_dir =  '../data/' + validation_data_name + '/'
		self.img_max = len(os.listdir(self.img_dir))
		self.data_iter  = data_iter

		self.Densenet.eval()
		for param in self.Densenet.parameters():
			param.requires_grad = False

		'''Chech Save Folser'''
		if self.img_dir == '../data/mpiigaze_validation/':
			filename =  "../result/mpiigaze_validation.txt" 

		elif self.img_dir == '../data/mpiigaze_personal_validation/':
			filename =  "../result/mpiigaze_personal_validation.txt" 

		elif self.img_dir == '../data/mpiigaze_test/':
			filename = '../result/mpiigaze_test.txt'

		elif self.img_dir == '../data/ut_multiview_validation/':
			filename = '../result/utmultiview_validation.txt' 

		elif self.img_dir == '../data/ut_multiview_test/':
			filename = '../result/utmultiview_test.txt' 

		elif self.img_dir == '../data/eth_xgaze_front_all/':
			filename = '../result/eth_xgaze_front_all.txt' 

		elif self.img_dir == '../data/columbia_all/':
			filename = '../result/columbia_all.txt' 

		elif self.img_dir == '../data/columbia_validation/':
			filename = '../result/columbia_validation.txt' 

		elif self.img_dir == '../data/columbia_test/':
			filename = '../result/columbia_test.txt' 

		elif self.img_dir == '../data/eth_xgaze_validation_left/':
			filename = '../result/eth_xgaze_validation_left.txt'
		
		elif self.img_dir == '../data/eth_xgaze_test_left/':
			filename = '../result/eth_xgaze_test_left.txt'

		elif self.img_dir == '../data/rt_gene_all/':
			filename = '../result/rt_gene_all.txt' 

		else:
			os._exit('Error self.img_dir name in sub_sim_gan')

		'''Make Result .txt file'''
		if not os.path.exists(filename):
			print('Make result.txt file')
			f = open(filename,'w')
			f.close()

		if self.img_max % config['test_batch_size']['size'] == 0:
			test_iter_num = int(self.img_max / config['test_batch_size']['size'])
		else:
			test_iter_num = int(self.img_max / config['test_batch_size']['size']) + 1

		for i in tqdm(range(test_iter_num)):
			test_img, gt_gaze, _ = next(self.data_iter)
				
			img_tensor = test_img.unsqueeze(0).cuda()
			img_tensor = torch.squeeze(img_tensor, dim=0)

			'''Predict gaze direction from images'''
			pred_gaze = self.Densenet(img_tensor)

			'''Normalize predicted gaze direction(3-dim)'''
			pred_gaze_normalized = torch.nn.functional.normalize(pred_gaze, dim=2)

			# '''Convert right-handed system'''
			# pred_gaze[:,1] = pred_gaze[:,1] * -1

			'''Revise extructed gt_gaze data from excel'''
			gt_gaze = np.array(gt_gaze)
			gt_gaze = gt_gaze.T	

			'''Deliete gt_gaze data number'''
			gt_gaze = gt_gaze[:,1:4]
			gt_gaze = torch.from_numpy(gt_gaze.astype(np.float32)).clone()

			'''Calculate difference of gaze direction'''
			diff_gaze = self.gaze_diff(gt_gaze, pred_gaze_normalized)

			diff_gaze_array = np.append(diff_gaze_array, diff_gaze)

			'''Vector Image'''
			test_img_1 = test_img[0]
			test_img_1_3ch = torch.cat((test_img_1, test_img_1, test_img_1), dim=0)
			test_img_1_3ch_pil = transforms.ToPILImage()(test_img_1_3ch.cpu())
			test_img_1_3ch_np = np.array(test_img_1_3ch_pil, dtype=np.uint8)
			test_img_1_3ch_cv2 = cv2.cvtColor(test_img_1_3ch_np, cv2.COLOR_RGB2BGR)
			test_img_1_3ch_cv2_vec = self.draw_gaze(test_img_1_3ch_cv2, gt_gaze, pred_gaze)
			test_img_1_3ch_pil = Image.fromarray(test_img_1_3ch_cv2_vec)
			test_img_1_tensor = transforms.ToTensor()(test_img_1_3ch_pil)
			test_img_1_tensor = test_img_1_tensor.unsqueeze(0)

			img_tensor_list.append(torch.unsqueeze(test_img_1_3ch, dim=0))
			img_tensor_list.append(test_img_1_tensor)
		
		stack_img = torch.cat(img_tensor_list, dim=0)
		make_grid_img = utils.make_grid(stack_img, 16)
		save_img_name = '../test_output/' + self.img_dir.split('/')[-2:-1][0] + '_' + str(self.current_step) + '.png'
		utils.save_image(make_grid_img, save_img_name)

		mean_diff_gaze = np.mean(diff_gaze_array)
		print('Validation Mean Error: ' + str(mean_diff_gaze))
		
		f = open(filename, 'a')
		f.write(str(mean_diff_gaze) + '\n')
		f.close()

		return mean_diff_gaze

