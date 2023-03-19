import torch
import cv2
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import json
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot_mat

def img_trans_coord(x, y, trans_value):
    
    original = 0
    trans_coord_x = random.uniform(original, trans_value)
    trans_coord_y = random.uniform(original, trans_value)

    return trans_coord_x, trans_coord_y

def move_ldmk_random_crop(moved_img, data_list, moved_x, moved_y):
    coord_list = []
    circle_img = moved_img.copy()
    # print('moved_x, moved_y: ',moved_x, moved_y)

    for i in range(len(data_list)):
        moved_x_coord = float(data_list[i][1]) + moved_x
        moved_y_coord = float(data_list[i][0]) + moved_y
        draw_x_coord, draw_y_coord = int(moved_x_coord), int(moved_y_coord)
        coord_list.append([moved_y_coord, moved_x_coord])

        '''Draw Landmark'''
        circle_img = cv2.circle(circle_img, (draw_x_coord, draw_y_coord), 3, (255,0,0),thickness = -1)

    # show_img(circle_img)
    return coord_list

def img_random_crop(ext_img, img_w, img_h, offset_x, offset_y):
    center_x = img_w/2
    center_y = img_h/2
    ext_center_x = int(ext_img.shape[1]/2)
    ext_center_y = int(ext_img.shape[0]/2)

    y_min = int(round(ext_center_y - (center_y + offset_y), 0))
    y_max = int(round(ext_center_y + center_y - offset_y, 0))
    x_min = int(round(ext_center_x - (center_x + offset_x), 0))
    x_max = int(round(ext_center_x + center_x - offset_x, 0))
    cropped_img = ext_img[y_min:y_max, x_min:x_max]
    # print('ext_center_y,center_y,offset_y: ',ext_center_y,center_y,offset_y)
    # print('ext_center_x, center_x, offset_x: ', ext_center_x, center_x, offset_x)
    # print('y_min,y_max,x_min,x_max: ',y_min,y_max,x_min,x_max)
    
    return cropped_img

def noise_value(range):
    min = range[0]
    max = range[1]
    value_range = abs(max - min)
    magnification = value_range / 150000
    value = range[0] + magnification
    # import pdb;pdb.set_trace()
    if value > range[1]:
        return random.uniform(0, value)
    else:
        return random.uniform(0, range[1])

def rotate_image(deg, img):
    w = img.shape[1]
    h = img.shape[0]
    mat = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    affine_img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return affine_img

def rot_ldmk(deg, ldmk_mtrx, x, y):
    img_center_x = x/2
    img_center_y = y/2
    ldmk_list = []
    rad = np.deg2rad(deg)

    for i in range(len(ldmk_mtrx)):
        ldmk_x = ldmk_mtrx[i][1] - img_center_x
        ldmk_y = ldmk_mtrx[i][0] - img_center_y
        ldmk_array = np.array([ldmk_y, ldmk_x])
        
        R = [[np.cos(rad), -np.sin(rad)],
             [np.sin(rad), np.cos(rad)]]
        
        rotated_ldmk = np.dot(R, ldmk_array)
        rotated_ldmk[1] = rotated_ldmk[1] + img_center_x
        rotated_ldmk[0] = rotated_ldmk[0] + img_center_y

        ldmk_list.append(rotated_ldmk)
    ldmk_out_array = np.array(ldmk_list)
    
    return ldmk_out_array

def rot_gaze(deg, gaze):
    gaze_array = np.array([gaze[0], gaze[1], gaze[2]])
    r = Rot_mat.from_euler('z', deg * -1, degrees=True)
    rotated_gaze_array = r.apply(gaze_array)

    return rotated_gaze_array

def gaussian_2d(w, h, cx, cy, sigma=2.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )
    #print("w =", w)
    #print("h =", h)

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=2.0))
  
    return np.array(heatmaps)

def save_img(img):
    img = img * 255
    img = img[:,:,np.newaxis]
    img = np.concatenate([img, img, img], axis = 2)
    cv2.imwrite('../check_img/error.jpg', img)
    # import pdb;pdb.set_trace()

def data_augmentation(imgs, gt_landmarks, gt_gazes, steps):
    
    augmentation_ranges = {  # (easy, hard)
        'translation': (2.0, 10.0),
        'rotation': (0.1, 5.0),
        'intensity': (1.1, 10.0),
        'blur': (0.1, 1.0),
        'scale': (0.01, 0.1),
        'rescale': (1.0, 0.2),
        'num_line': (0.0, 2.0),
        'heatmap_sigma': (5.0, 2.5),
    }

    '''Image Shape'''
    img_w = imgs.shape[3]
    img_h = imgs.shape[2]

    '''Output'''
    output_img_list = []
    heatmaps_list = []
    landmarks_list = []
    gt_gaze_list = []

    # imgs = torch.cat((imgs, imgs, imgs), dim=1)
    imgs = imgs.permute(0, 2, 3, 1)
    imgs_array = imgs.to('cpu').detach().numpy().copy()
    gt_landmarks_array = gt_landmarks.to('cpu').detach().numpy().copy()
    gt_gazes_array = gt_gazes.to('cpu').detach().numpy().copy()

    for i in range(imgs.shape[0]):
        eye_img = imgs_array[i]
        gt_ldmks = gt_landmarks_array[i]
        gt_gaze =  gt_gazes_array[i]
        gt_gaze = np.squeeze(gt_gaze)
        # show_img(eye_img)
        # draw_circle_image(eye_img, gt_ldmks)
        # show_gaze_img(eye_img, gt_gaze, gt_ldmks)

        '''decide noise value'''
        translation_noise = noise_value(augmentation_ranges['translation'])
        rotation_noise = noise_value(augmentation_ranges['rotation'])
        intensity_noise = noise_value(augmentation_ranges['intensity'])
        blur_noise = noise_value(augmentation_ranges['blur'])
        scale_noise = noise_value(augmentation_ranges['scale'])
        # rescale_max = noise_value(augmentation_ranges['rescale'])
        num_line_noise = noise_value(augmentation_ranges['num_line'])
        heatmap_sigma_noise = noise_value(augmentation_ranges['heatmap_sigma'])

        '''Translation'''
        if translation_noise > 0:
            img_transed_xcoord, img_transed_ycoord = img_trans_coord(eye_img.shape[1], eye_img.shape[0], translation_noise)
            # resize_x = eye_img.shape[0] + 10*2
            # resize_ratio = resize_x / eye_img.shape[0]
            # resize_y = int(eye_img.shape[1] * resize_ratio)
            # extended_eye_img = cv2.resize(eye_img, (resize_y,resize_x))
            extended_eye_img = np.pad(eye_img, ((10, 10), (10, 10), (0, 0)),'edge')
            eye_img = img_random_crop(extended_eye_img, eye_img.shape[1], eye_img.shape[0], img_transed_xcoord, img_transed_ycoord)
            gt_ldmks = move_ldmk_random_crop(eye_img, gt_ldmks, img_transed_xcoord, img_transed_ycoord)
            # print('Translation Noise: ',translation_noise)
            # show_img(eye_img)
            # draw_circle_image(eye_img, gt_ldmks)

        '''Rotation'''
        if rotation_noise > 0:
            # print('rotation_noise: ',rotation_noise)
            eye_img = rotate_image(rotation_noise, eye_img)
            gt_ldmks = rot_ldmk(rotation_noise, gt_ldmks, img_w, img_h)
            gt_gaze = rot_gaze(rotation_noise, gt_gaze)
            # show_img(eye_img)
            # draw_circle_image(eye_img, gt_ldmks)

        '''Draw line randomly'''
        if num_line_noise > 0:
            line_rand_nums = np.random.rand(int(5 * num_line_noise))
            for i in range(int(num_line_noise)):
                j = 5 * i
                lx0, ly0 = int(img_w * line_rand_nums[j]), img_h
                lx1, ly1 = img_w, int(img_h * line_rand_nums[j + 1])
                direction = line_rand_nums[j + 2]
                if direction < 0.25:
                    lx1 = ly0 = 0
                elif direction < 0.5:
                    lx1 = 0
                elif direction < 0.75:
                    ly0 = 0
                line_colour = line_rand_nums[j + 3]
                eye_img = cv2.line(eye_img, (lx0, ly0), (lx1, ly1),
                              color=(line_colour, line_colour, line_colour),
                            #   thickness=1,
                              thickness=max(1, int(6*line_rand_nums[j + 4])),
                              lineType=cv2.LINE_AA)
                # print('num_line_noise: ',num_line_noise)
                # show_img(eye_img)

        '''Rescale'''
        value_range = abs(augmentation_ranges['rescale'][0] - augmentation_ranges['rescale'][1])
        magnification = value_range / 600000
        # low_value = augmentation_ranges['rescale'][0] - (magnification * steps)
        low_value = 0.2
        rescale_noise = np.random.uniform(low=low_value, high=1.0)
        interpolation = cv2.INTER_CUBIC
        # save_img(eye_img)
        # print('eye_img.shape: ', eye_img.shape)
        # print('rescale_noise: ', rescale_noise)
        eye_img = cv2.resize(eye_img, dsize=(0, 0), fx=rescale_noise, fy=rescale_noise, interpolation=interpolation)
        # eye_img = cv2.resize(eye_img, None, fx=rescale_noise, fy=rescale_noise, interpolation=interpolation)
        eye_img = cv2.normalize(eye_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
        eye_img = cv2.equalizeHist(eye_img)
        eye_img = eye_img.astype(np.float32)
        eye_img *= 2.0 / 255.0
        eye_img -= 1.0
        # print('normalized: ',eye_img)
        eye_img = cv2.resize(eye_img, dsize=(img_w, img_h), interpolation=interpolation)
        # print('Rescale noise: ',rescale_max)
        # show_img(eye_img)
        # draw_circle_image(eye_img, gt_ldmks)

        '''Intensity'''
        # Add rgb noise to eye image
        if intensity_noise > 1.0:
            # print('Intensity Noise:', intensity_noise)
            eye_img += np.random.randint(low=-intensity_noise, high=intensity_noise,
                                     size=eye_img.shape, dtype=np.int32)/255
            cv2.normalize(eye_img, eye_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # show_img(eye_img)

        '''Add blur to eye image'''
        if blur_noise > 0:
            eye_img = cv2.GaussianBlur(eye_img, (7, 7), 0.5 + np.abs(blur_noise)) #karnel 7x7, standard deviation:0.5 + noise
            # print('Gaussian Blur Noise', blur_noise)
            # show_img(eye_img)
            # draw_circle_image(eye_img, gt_ldmks)
            # show_gaze_img(eye_img, gt_gaze, gt_ldmks)

        landmarks = np.array(gt_ldmks)
        temp = np.zeros((34, 2), dtype=np.float32)
        temp[:, 0] = landmarks[:, 0]
        temp[:, 1] = landmarks[:, 1]
        landmarks = temp
        heatmaps = get_heatmaps(w=img_w, h=img_h, landmarks=landmarks)
        gt_heatmaps = np.asarray(heatmaps)
        assert heatmaps.shape == (34, img_h, img_w)
        # show_heatmap(gt_heatmaps, eye_img)
        
        eye_img = eye_img[:,:,np.newaxis,np.newaxis]
        eye_img = eye_img.transpose(2,3,0,1)
        output_img = torch.from_numpy(eye_img.astype(np.float32)).clone()

        heatmaps = torch.from_numpy(heatmaps.astype(np.float32)).clone().cuda()
        heatmaps = torch.unsqueeze(heatmaps,dim=0).cuda()

        landmarks = torch.from_numpy(landmarks.astype(np.float32)).clone().cuda()
        landmarks = torch.unsqueeze(landmarks,dim=0).cuda()

        gt_gaze = torch.from_numpy(gt_gaze.astype(np.float32)).clone().cuda()
        gt_gaze = torch.unsqueeze(gt_gaze,dim=0).cuda()

        output_img_list.append(output_img)
        heatmaps_list.append(heatmaps)
        landmarks_list.append(landmarks)
        gt_gaze_list.append(gt_gaze)

    output_img_tensor = torch.cat(output_img_list)
    heatmaps_tensor = torch.cat(heatmaps_list)
    landmarks_tensor = torch.cat(landmarks_list)
    gt_gaze_tensor = torch.cat(gt_gaze_list)
    gt_gaze_tensor = torch.unsqueeze(gt_gaze_tensor, dim=1)

    return output_img_tensor, heatmaps_tensor, landmarks_tensor, gt_gaze_tensor

    

def add_blur(imgs, steps):
    
    blur_range = {  # (easy, hard)
        'translation': (2.0, 10.0),
        'rotation': (0.1, 5.0),
        'intensity': (1.1, 10.0),
        'blur': (0.1, 1.0),
        'scale': (0.01, 0.1),
        'rescale': (1.0, 0.2),
        'num_line': (0.0, 2.0),
        'heatmap_sigma': (5.0, 2.5),
    }


    '''Image Shape'''
    img_w = imgs.shape[3]
    img_h = imgs.shape[2]

    '''Output'''
    output_img_list = []
    heatmaps_list = []
    landmarks_list = []
    gt_gaze_list = []

    imgs = imgs.permute(0, 2, 3, 1)
    imgs_array = imgs.to('cpu').detach().numpy().copy()
    # import pdb;pdb.set_trace()

    for i in range(imgs.shape[0]):
        eye_img = imgs_array[i]

        '''decide blur value'''
        intensity_noise = noise_value(blur_range['intensity'])
        blur_noise = noise_value(blur_range['blur'])
        scale_noise = noise_value(blur_range['scale'])

        value_range = abs(blur_range['rescale'][0] - blur_range['rescale'][1])
        magnification = value_range / 10000
        # low_value = blur_range['rescale'][0] - (magnification * steps)
        low_value = 0.2
        rescale_noise = np.random.uniform(low=low_value, high=1.0)
        interpolation = cv2.INTER_CUBIC
        # import pdb;pdb.set_trace()
        eye_img = cv2.resize(eye_img, dsize=(0, 0), fx=rescale_noise, fy=rescale_noise, interpolation=interpolation)
        eye_img = cv2.normalize(eye_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX).astype(np.uint8)
        eye_img = cv2.equalizeHist(eye_img)
        eye_img = eye_img.astype(np.float32)
        eye_img *= 2.0 / 255.0
        eye_img -= 1.0
        eye_img = cv2.resize(eye_img, dsize=(img_w, img_h), interpolation=interpolation)

        '''Intensity'''
        # Add rgb noise to eye image
        if intensity_noise > 1.0:
            eye_img += np.random.randint(low=-intensity_noise, high=intensity_noise,
                                     size=eye_img.shape, dtype=np.int32)/255
            cv2.normalize(eye_img, eye_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        '''Add blur to eye image'''
        if blur_noise > 0:
            eye_img = cv2.GaussianBlur(eye_img, (7, 7), 0.5 + np.abs(blur_noise)) #karnel 7x7, standard deviation:0.5 + noise

        eye_img = eye_img[:,:,np.newaxis,np.newaxis]
        eye_img = eye_img.transpose(2,3,0,1)
        output_img = torch.from_numpy(eye_img.astype(np.float32)).clone()
        output_img_list.append(output_img)

    output_img_tensor = torch.cat(output_img_list)

    return output_img_tensor