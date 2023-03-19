import scipy.io
import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
import cv2
from tqdm import tqdm
import json
import math
import torch
from torchvision import transforms
from torchvision import utils
import torch.nn as nn
import yaml

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)


def find_center(img, data_list, flag = 1):

    """Find x,y-max & min"""
    x_list = data_list[:,:,1]
    x_max_idx = torch.max(x_list, 1)
    x_min_idx = torch.min(x_list, 1)

    y_list = data_list[:,:,0]
    y_max_idx = torch.max(y_list, 1)
    y_min_idx = torch.min(y_list, 1)

    idx_sub = torch.arange(data_list.shape[0])

    x_max = data_list[idx_sub, x_max_idx[1], :]
    x_min = data_list[idx_sub, x_min_idx[1], :]
    y_max = data_list[idx_sub, y_max_idx[1], :]
    y_min = data_list[idx_sub, y_min_idx[1], :]
    
    """Find Corner"""
    right_corner = x_max
    left_corner = x_min
          
    """Find Top & Bottom"""    
    top = y_min
    bottom = y_max
    
    """Find Center Point"""
    center_x = (right_corner[:,1] + left_corner[:,1]) / 2
    center_y = (top[:,0] + bottom[:,0]) / 2
    center = torch.stack([center_x, center_y], dim = 1)

    """Calculate Radius"""
    bias = int(img.shape[3]*0.05)
    
    if flag == 1:   
        radius_x = right_corner[:,1] - center[:,0] + bias
        radius_y = bottom[:,0] - center[:,1] + bias
        
    else:
        radius_x = right_corner[:,1] - center[:,0] - (bias * 0.5)
        radius_y = bottom[:,0] - center[:,1] - (bias * 0.5)

    """Calculate Tilt"""
    vec_std = torch.tensor([1,0])
    vec_std = vec_std.expand(center.shape[0],2).cuda()
    vec_eye = right_corner - left_corner
    vec_eye = vec_eye.cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    tilt = cos(vec_std, vec_eye)

    # print("rad_x:",radius_x, " rad_y:", radius_y)
    # import pdb;pdb.set_trace()

    return center, radius_x, radius_y, tilt

def make_tilt_mask(img, center, rad_x, rad_y, tilt):
    x_map = torch.arange(img.shape[3]).reshape(1,-1)
    x_map = x_map.expand(img.shape[2], img.shape[3])
    x_map = torch.unsqueeze(x_map, 0)
    x_map = x_map.repeat(img.shape[0], 1, 1, 1).cuda()
    
    y_map = torch.arange(img.shape[2]).reshape(-1,1)
    y_map = img.shape[2] - y_map
    y_map = y_map.expand(img.shape[2], img.shape[3])
    y_map = torch.unsqueeze(y_map, 0)
    y_map = y_map.repeat(img.shape[0], 1, 1, 1).cuda()
    
    x0 = center[:, 0]
    x0 = torch.reshape(x0, (img.shape[0], 1))
    x0 = torch.unsqueeze(x0, 2)
    x0 = torch.unsqueeze(x0, 2).cuda()
    
    y0 = img.shape[2] - center[:,1]
    y0 = torch.reshape(y0, (img.shape[0], 1))
    y0 = torch.unsqueeze(y0, 2)
    y0 = torch.unsqueeze(y0, 2).cuda()

    cos = torch.cos(tilt)
    cos = torch.reshape(cos, (img.shape[0], 1))
    cos = torch.unsqueeze(cos, 2)
    cos = torch.unsqueeze(cos, 2).cuda()
    
    sin = torch.sin(tilt)
    sin = torch.reshape(sin, (img.shape[0], 1))
    sin = torch.unsqueeze(sin, 2)
    sin = torch.unsqueeze(sin, 2).cuda()

    rad_x = torch.reshape(rad_x, (img.shape[0], 1))
    rad_x = torch.unsqueeze(rad_x, 2)
    rad_x = torch.unsqueeze(rad_x, 2).cuda()
    
    rad_y = torch.reshape(rad_y, (img.shape[0], 1))
    rad_y = torch.unsqueeze(rad_y, 2)
    rad_y = torch.unsqueeze(rad_y, 2).cuda()
    
    mask = ((((x_map - x0) * cos) + ((y_map - y0) * sin)) **2 / (rad_x) ** 2) + ((((-1 * (x_map - x0) * sin) + (y_map - y0) * cos) ** 2) / (rad_y) ** 2) - 1
    mask_img = torch.where(mask < 0, 1, 0)

    return mask_img
    

def mask_generator(img, eyelids, iris):
    """Find eyelids center"""
    eyelids_center, eyelids_rad_x, eyelids_rad_y, eyelids_tilt = find_center(img, eyelids)

    """Make Eyelids Mask"""
    eyelids_mask = make_tilt_mask(img, eyelids_center, eyelids_rad_x, eyelids_rad_y, eyelids_tilt)

    """Find iris center"""
    iris_center, iris_rad_x, iris_rad_y, iris_tilt = find_center(img, iris, flag = 0)

    """Make Iris Mask"""
    iris_mask = make_tilt_mask(img, iris_center, iris_rad_x, iris_rad_y, iris_tilt)
    iris_mask = torch.where(iris_mask == 0, 1, 0)

    """Make Mask"""
    mask = eyelids_mask * iris_mask

    # import pdb;pdb.set_trace()

    return mask