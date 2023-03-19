import os
import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import torch.nn.functional as F
import yaml
from torchvision import transforms,utils, models
import math
import tqdm
import csv
import random
import yaml
from torchvision.utils import save_image
from util.data_augmentator import data_augmentation
from PIL import Image, ImageFilter
from util.gazemap_tensor import gazemap_generator

with open('config.yaml', 'r') as yml:
    config = yaml.load(yml,Loader=yaml.Loader)

def eval_synthe_gazemap(Reg, synthe_eval_data_iter, current_step):
    stack_img = torch.empty(7, 3, 36, 60)
    save_dir = config['validation_ldmk']['dirname']
    synthe_img, _, gt_landmarks, _ = next(synthe_eval_data_iter)
    _, pred_ldmks = Reg(synthe_img)
    gt_gazemap = gazemap_generator(synthe_img, gt_landmarks)
    pred_gaze_maps = gazemap_generator(synthe_img, pred_ldmks)
    # pred_gaze = Densenet(pred_gaze_map)

    for i in range(synthe_img.shape[0]):
        synthetic_img = synthe_img[i]
        synthetic_img_3ch = torch.cat((synthetic_img, synthetic_img, synthetic_img), dim=0)
        synthetic_img_pil = transforms.ToPILImage()(synthetic_img_3ch.cpu())
        synthetic_img_np = np.array(synthetic_img_pil, dtype=np.uint8)
        synthetic_img_cv2 = cv2.cvtColor(synthetic_img_np, cv2.COLOR_RGB2BGR)
        synthetic_img_pil = Image.fromarray(synthetic_img_cv2)
        synthetic_img_tensor = transforms.ToTensor()(synthetic_img_pil)
        synthetic_img_tensor = synthetic_img_tensor.unsqueeze(0)        

        pred_gaze_map_2ch = pred_gaze_maps[i]
        pred_gaze_map_2ch_1 = pred_gaze_map_2ch[0]
        pred_gaze_map_2ch_2 = pred_gaze_map_2ch[1]
        pred_gaze_map_1ch = pred_gaze_map_2ch_2 - pred_gaze_map_2ch_1
        pred_gaze_map_1ch = torch.unsqueeze(pred_gaze_map_1ch, dim=0)
        pred_gaze_map_3ch = torch.cat((pred_gaze_map_1ch, pred_gaze_map_1ch, pred_gaze_map_1ch), dim=0)
        pred_gaze_map_3ch = torch.unsqueeze(pred_gaze_map_3ch, dim=0)

        pred_gaze_map_2ch_1 = torch.unsqueeze(pred_gaze_map_2ch_1, dim=0)
        pred_gaze_map_2ch_1_3ch = torch.cat((pred_gaze_map_2ch_1, pred_gaze_map_2ch_1, pred_gaze_map_2ch_1), dim=0)
        pred_gaze_map_2ch_1_3ch = torch.unsqueeze(pred_gaze_map_2ch_1_3ch, dim=0)

        pred_gaze_map_2ch_2 = torch.unsqueeze(pred_gaze_map_2ch_2, dim=0)
        pred_gaze_map_2ch_2_3ch = torch.cat((pred_gaze_map_2ch_2, pred_gaze_map_2ch_2, pred_gaze_map_2ch_2), dim=0)
        pred_gaze_map_2ch_2_3ch = torch.unsqueeze(pred_gaze_map_2ch_2_3ch, dim=0)

        gt_gaze_map_2ch = gt_gazemap[i]
        gt_gaze_map_2ch_1 = gt_gaze_map_2ch[0]
        gt_gaze_map_2ch_2 = gt_gaze_map_2ch[1]
        gt_gaze_map_1ch = gt_gaze_map_2ch_2 - gt_gaze_map_2ch_1
        gt_gaze_map_1ch = torch.unsqueeze(gt_gaze_map_1ch, dim=0)
        gt_gaze_map_3ch = torch.cat((gt_gaze_map_1ch, gt_gaze_map_1ch, gt_gaze_map_1ch), dim=0)
        gt_gaze_map_3ch = torch.unsqueeze(gt_gaze_map_3ch, dim=0)

        gt_gaze_map_2ch_1 = torch.unsqueeze(gt_gaze_map_2ch_1, dim=0)
        gt_gaze_map_2ch_1_3ch = torch.cat((gt_gaze_map_2ch_1, gt_gaze_map_2ch_1, gt_gaze_map_2ch_1), dim=0)
        gt_gaze_map_2ch_1_3ch = torch.unsqueeze(gt_gaze_map_2ch_1_3ch, dim=0)

        gt_gaze_map_2ch_2 = torch.unsqueeze(gt_gaze_map_2ch_2, dim=0)
        gt_gaze_map_2ch_2_3ch = torch.cat((gt_gaze_map_2ch_2, gt_gaze_map_2ch_2, gt_gaze_map_2ch_2), dim=0)
        gt_gaze_map_2ch_2_3ch = torch.unsqueeze(gt_gaze_map_2ch_2_3ch, dim=0)

        # import pdb;pdb.set_trace()  
        stack_img = torch.cat((stack_img.cuda(), synthetic_img_tensor.cuda(), gt_gaze_map_3ch.cuda(), gt_gaze_map_2ch_1_3ch.cuda(), gt_gaze_map_2ch_2_3ch.cuda(), pred_gaze_map_3ch.cuda(), pred_gaze_map_2ch_1_3ch.cuda(), pred_gaze_map_2ch_2_3ch.cuda()), dim = 0)

    '''Save Image''' 
    make_grid_all_images = utils.make_grid(stack_img, 7)
    utils.save_image(make_grid_all_images, save_dir + 'synthe_gazemap_' + str(current_step) + ".jpg")

def eval_synthe_img(R1_past, Reg, real_data_iter, eval_data_iter, current_step):
    real_images = next(real_data_iter)
    '''eval synthe img'''    
    clamp_synthetic_images, clamp_gt_heatmaps, clamp_gt_landmarks, clamp_gt_gaze = next(eval_data_iter)
    '''eval randmom synthe img'''
    # clamp_synthetic_images, clamp_gt_heatmaps, clamp_gt_landmarks, clamp_gt_gaze = next(self.synthetic_data_iter)
    save_refined_images = R1_past(clamp_synthetic_images.cuda())
    save_refined_images = torch.sigmoid(save_refined_images)
    # save_refined_images, clamp_gt_heatmaps, clamp_gt_landmarks, clamp_gt_gaze = data_augmentation(save_refined_images, clamp_gt_landmarks, clamp_gt_gaze)	
    save_refined_images = torch.clamp(save_refined_images, min=0, max=1)	
    save_refined_heatmaps_pred, save_refined_landmarks_pred = Reg(save_refined_images.cuda())

    stack_real_images = torch.empty(0, 3, 36, 60)
    stack_synthetic_images = torch.empty(0, 3, 36, 60)
    stack_eye_blend_images = torch.empty(0, 3, 36, 60)
    stack_refined_images = torch.empty(0, 3, 36, 60)
    stack_refined_blend_images = torch.empty(0, 3, 36, 60)
    stack_all_images = torch.empty(5, 3, 36, 60)

    for i in range(config['eval_batch_size']['size']):
        save_synthetic_img = torch.empty(1, 36, 60)
        save_refined_img = torch.empty(1, 36, 60)
        
        transform_img = transforms.Compose([
            transforms.ToPILImage()
        ])
        transform_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

        save_synthetic_img = clamp_synthetic_images[i].cuda()
        gt_hm = clamp_gt_heatmaps[i].cuda()
        gt_hm = np.mean(gt_hm[0:32].cpu().detach().numpy(), axis=0)
        gt_hm = cv2.normalize(gt_hm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite('gt_heatmap.png', gt_hm*255)
        gt_hm_color = cv2.imread('gt_heatmap.png')
        gt_hm_color = cv2.applyColorMap(gt_hm_color, cv2.COLORMAP_JET)

        save_synthetic_img_3ch = torch.cat((save_synthetic_img, save_synthetic_img, save_synthetic_img), dim=0)
        utils.save_image(save_synthetic_img_3ch, 'synthetic_3ch_tensor.png')
        save_synthetic_img_3ch_pil = transforms.ToPILImage()(save_synthetic_img_3ch.cpu())
        save_synthetic_img_3ch_np = np.asarray(save_synthetic_img_3ch_pil, dtype=np.uint8)
        save_synthetic_img_3ch_np = cv2.cvtColor(save_synthetic_img_3ch_np, cv2.COLOR_RGB2BGR)

        alpha = 0.3
        blended_image = cv2.addWeighted(gt_hm_color, alpha, save_synthetic_img_3ch_np, 1-alpha, 0)

        blended_image_cv2 = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        blended_image_pil = Image.fromarray(blended_image_cv2)
        eye_blend_images = transform_tensor(blended_image_pil)

        eye_blend_images = eye_blend_images.unsqueeze(0)

        save_refined_img = save_refined_images[i].cuda()
        save_real_img = real_images[i].cuda()
        save_heatmaps_img = save_refined_heatmaps_pred[i].cuda()
        save_heatmaps_img = save_heatmaps_img.squeeze(0)
        hm_pred = np.mean(save_heatmaps_img[-1, 0:32].cpu().detach().numpy(), axis=0)
        # import pdb;pdb.set_trace()
        hm_pred = cv2.normalize(hm_pred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite('pred.png', hm_pred*255)
        hm_pred_color = cv2.imread('pred.png')
        
        i = 0
        while i <= 35:
            j = 0
            while j <= 59:
                #print(j)
                if hm_pred_color[i][j][0] < 70 or hm_pred_color[i][j][1] < 70 or hm_pred_color[i][j][2] < 70:
                    hm_pred_color[i][j][0]=0
                    hm_pred_color[i][j][1]=0
                    hm_pred_color[i][j][2]=0
                    j = j + 1
                else:
                    j = j + 1
            i = i + 1
        
        hm_pred_color = cv2.applyColorMap(hm_pred_color, cv2.COLORMAP_JET)
        save_refined_img_3ch = torch.cat((save_refined_img, save_refined_img, save_refined_img), dim=0)
        save_real_img_3ch = torch.cat((save_real_img, save_real_img, save_real_img), dim = 0)
        save_refined_img_pil = transforms.ToPILImage()(save_refined_img_3ch.cpu())
        # save_real_img_pil = transforms.ToPILImage()(save_real_img_3ch.cpu())
        save_refined_img_np = np.asarray(save_refined_img_pil, dtype=np.uint8)
        # save_real_img_np = np.array(save_real_img_pil, dtype=np.uint8)
        save_refined_img_np = cv2.cvtColor(save_refined_img_np, cv2.COLOR_RGB2BGR)
        # save_real_img_np = cv2.cvtColor(save_real_img_np, cv2.COLOR_RGB2BGR)
        alpha = 0.3
        refined_blended_image = cv2.addWeighted(hm_pred_color, alpha, save_refined_img_np, 1-alpha, 0)
        refined_blended_image_cv2 = cv2.cvtColor(refined_blended_image, cv2.COLOR_BGR2RGB)
        refined_blended_image_pil = Image.fromarray(refined_blended_image_cv2)
        refined_blend_images = transforms.ToTensor()(refined_blended_image_pil)
        refined_blend_images = refined_blend_images.unsqueeze(0)
        save_synthetic_img_3ch = save_synthetic_img_3ch.unsqueeze(0)
        save_refined_img_3ch = save_refined_img_3ch.unsqueeze(0)
        save_real_img_3ch = save_real_img_3ch.unsqueeze(0)
        # import pdb;pdb.set_trace()
        stack_all_images = torch.cat((stack_all_images.cuda(), save_real_img_3ch.cuda(), save_synthetic_img_3ch.cuda(), save_refined_img_3ch.cuda(), eye_blend_images.cuda(), refined_blend_images.cuda()), dim = 0)
        # stack_all_images = torch.cat((save_real_img_3ch.cuda(), save_synthetic_img_3ch.cuda(), eye_blend_images.cuda(), save_refined_img_3ch.cuda(), refined_blend_images.cuda()), dim = 0)
        i = i + 1
    make_grid_all_images = utils.make_grid(stack_all_images, 5)
    
    utils.save_image(make_grid_all_images, config['save_image_directory']['dirname'] + 'all_' + str(current_step) + '.png')
    # """

# def eval_real_img_landmark(Reg, Gaze, real_eval_data_iter, current_step):
def eval_real_img_landmark(ensenet, real_eval_data_iter, current_step):
    stack_img = torch.empty(25, 3, 36, 60)
    ldmk_img_list = []
    save_dir = config['validation_ldmk']['dirname']
    # real_img = next(self.real_data_iter)
    real_img = next(real_eval_data_iter)
    # heatmaps_pred, landmarks_pred = Reg(real_img)
    # gazemaps_pred = gazemap_generator(real_img, landmarks_pred)
    # gaze_pred = Gaze(landmarks_pred)
    gaze_pred = Densenet(real_img)
    gazemaps_pred_1ch = gazemaps_pred[:,1] - gazemaps_pred[:,0]
    gazemaps_pred_1ch = torch.unsqueeze(gazemaps_pred_1ch,dim=1)

    for i in range(real_img.shape[0]):
        save_real_img = real_img[i].cuda()
        save_real_img_3ch = torch.cat((save_real_img, save_real_img, save_real_img), dim = 0)
        save_real_img_pil = transforms.ToPILImage()(save_real_img_3ch.cpu())
        save_real_img_np = np.asarray(save_real_img_pil, dtype=np.uint8)
        save_real_img_cv2 = cv2.cvtColor(save_real_img_np, cv2.COLOR_RGB2BGR)
        gazemaps_pred_3ch = torch.cat((torch.unsqueeze(gazemaps_pred_1ch[i], 0), torch.unsqueeze(gazemaps_pred_1ch[i], 0), torch.unsqueeze(gazemaps_pred_1ch[i], 0)), dim=1) 

        '''draw landmark'''
        ldmk_img = draw_ldmk(save_real_img_cv2, landmarks_pred[i], current_step)
        ldmk_img = draw_gaze(ldmk_img, landmarks_pred[i], gaze_pred[i])
        # ldmk_img_list.append(ldmk_img)
        ldmk_img_pil = Image.fromarray(ldmk_img)
        ldmk_img_tensor = transforms.ToTensor()(ldmk_img_pil)
        ldmk_img_tensor = ldmk_img_tensor.unsqueeze(0)
        
        save_real_img_3ch = torch.unsqueeze(save_real_img_3ch, dim=0)
        stack_img = torch.cat((stack_img.cuda(), save_real_img_3ch.cuda(), ldmk_img_tensor.cuda(), gazemaps_pred_3ch.cuda()))
    make_grid_all_images = utils.make_grid(stack_img, 30)
    # ldmk_img_array = np.array(ldmk_img_list).reshape(-1,60,3) 
    # cv2.imwrite(save_dir + "cv2_" + str(self.current_step) + ".jpg",ldmk_img_array)
    utils.save_image(make_grid_all_images, save_dir + 'real_' + str(current_step) + ".jpg")
    # import pdb;pdb.set_trace()

# def eval_synthe_img_landmark(R1, Reg, Gaze, synthe_eval_data_iter):
def eval_synthe_img_landmark(R1, Reg, Densenet, synthe_eval_data_iter, current_step):
    stack_img = torch.empty(5, 3, 36, 60)
    gt_stack_img = torch.empty(5, 3, 36, 60)
    ldmk_img_list = []
    save_dir = config['validation_ldmk']['dirname']
    synthe_img, _, gt_ldmk, gt_gaze = next(synthe_eval_data_iter)
    refine_img = R1(synthe_img)
    refine_img = torch.sigmoid(refine_img)
    refine_img = torch.clamp(refine_img, min = 0, max = 1)
    refine_aug_img, _, gt_aug_ldmk, gt_aug_gaze = data_augmentation(refine_img, gt_ldmk, gt_gaze)
    heatmaps_pred, landmarks_pred = Reg(refine_img)
    gazemaps_pred = gazemap_generator(refine_img, landmarks_pred)
    gaze_pred = Densenet(gazemaps_pred)
    for i in range(synthe_img.shape[0]):
        save_img = refine_aug_img[i].cuda()
        save_img_3ch = torch.cat((save_img, save_img, save_img), dim = 0)
        save_img_pil = transforms.ToPILImage()(save_img_3ch.cpu())
        save_img_np = np.asarray(save_img_pil, dtype=np.uint8)
        save_img_cv2 = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
        '''draw landmark'''
        ldmk_img = draw_ldmk(save_img_cv2, landmarks_pred[i], current_step)
        ldmk_img = draw_gaze(ldmk_img, landmarks_pred[i], gaze_pred[i])
        # ldmk_img_list.append(ldmk_img)
        ldmk_img_pil = Image.fromarray(ldmk_img)
        ldmk_img_tensor = transforms.ToTensor()(ldmk_img_pil)
        ldmk_img_tensor = ldmk_img_tensor.unsqueeze(0)
        stack_img = torch.cat((stack_img.cuda(), ldmk_img_tensor.cuda()))
        '''If Save GT Data'''
        save_aug_img = refine_aug_img[i].cuda()
        save_aug_img_3ch = torch.cat((save_aug_img, save_aug_img, save_aug_img), dim = 0)
        save_aug_img_pil = transforms.ToPILImage()(save_aug_img_3ch.cpu())
        save_aug_img_np = np.asarray(save_aug_img_pil, dtype=np.uint8)
        save_aug_img_cv2 = cv2.cvtColor(save_aug_img_np, cv2.COLOR_RGB2BGR)
        gt_ldmk_img = draw_ldmk(save_aug_img_cv2, gt_aug_ldmk[i], current_step)
        gt_gaze_img = draw_gaze(gt_ldmk_img, gt_aug_ldmk[i], torch.unsqueeze(gt_aug_gaze[i],0))
        # gt_ldmk_img_pil = Image.fromarray(gt_ldmk_img)
        gt_gaze_img_pil = Image.fromarray(gt_gaze_img)
        # gt_ldmk_img_tensor = transforms.ToTensor()(gt_ldmk_img_pil)
        gt_gaze_img_tensor = transforms.ToTensor()(gt_gaze_img_pil)
        # gt_ldmk_img_tensor = gt_ldmk_img_tensor.unsqueeze(0)
        gt_gaze_img_tensor = gt_gaze_img_tensor.unsqueeze(0)
        # gt_stack_img = torch.cat((gt_stack_img.cuda(), gt_ldmk_img_tensor.cuda(), gt_gaze_img_tensor.cuda()),dim = 0)
        gt_stack_img = torch.cat((gt_stack_img.cuda(), gt_gaze_img_tensor.cuda()),dim = 0)

    '''Save Image''' 
    make_grid_all_images = utils.make_grid(stack_img, 5)
    make_grid_all_gt_images = utils.make_grid(gt_stack_img, 5)
    utils.save_image(make_grid_all_images, save_dir + 'synthe_' + str(current_step) + ".jpg")
    utils.save_image(make_grid_all_gt_images, save_dir + 'gt_' + str(current_step) + '.jpg')

def draw_gaze(img, ldmk, gaze):
    iris_center_x = ldmk[32][1]
    iris_center_y = ldmk[32][0]
    gaze_x = gaze[0][0]
    gaze_y = gaze[0][1]
    arrow_x = iris_center_x + (50 * gaze_x)
    arrow_y = iris_center_y + (50 * gaze_y)
    img = cv2.arrowedLine(img, (int(iris_center_x), int(iris_center_y)), (int(arrow_x), int(arrow_y)), (255, 0, 0), thickness=1)

    return img

def draw_ldmk(img, ldmk, current_step):
    save_img = None
    cnt = current_step
    save_img = img
    # save_img = img.repeat(3,1,1)
    # save_img = save_img.permute(1,2,0).cpu().numpy().copy()

    eyelids_ldmk = ldmk[0:16]
    iris_ldmk = ldmk[16:32]
    center_ldmk = ldmk[32:33]
    eye_ball_ldmk = ldmk[33]

    # import pdb;pdb.set_trace()

    for n_landmarks in range(eyelids_ldmk.shape[0]):
        save_img = cv2.circle(save_img, (int(eyelids_ldmk[n_landmarks][1]),int(eyelids_ldmk[n_landmarks][0])), 1, (0,255,0), thickness=-1)

    for n_landmarks in range(iris_ldmk.shape[0]):
        save_img = cv2.circle(save_img, (int(iris_ldmk[n_landmarks][1]),int(iris_ldmk[n_landmarks][0])), 1, (0,0,255), thickness=-1)
    
    save_img = cv2.circle(save_img, (int(center_ldmk[0][1]),int(center_ldmk[0][0])), 1, (255,0,0), thickness=-1)
    save_img = cv2.circle(save_img, (int(eye_ball_ldmk[1]),int(eye_ball_ldmk[0])), 1, (255,255,255), thickness=-1)

    return save_img

def confirm_ldmk(R1, synthe_eval_data_iter, current_step):
    synthe_imgs, _, landmarks, _, _ = next(synthe_eval_data_iter)
    refine_imgs = R1(synthe_imgs)
    stack_img = torch.empty(5, 3, 36, 60)
    ldmk_img_list = []
    save_dir = config['validation_ldmk']['dirname']
    for i in range(refine_imgs.shape[0]):
        save_img = torch.clamp(refine_imgs[i].cuda(), min=0, max=1)
        # save_img = torch.sigmoid(refine_imgs[i])
        save_img_3ch = torch.cat((save_img, save_img, save_img), dim = 0)
        save_img_pil = transforms.ToPILImage()(save_img_3ch.cpu())
        save_img_np = np.asarray(save_img_pil, dtype=np.uint8)
        save_img_cv2 = cv2.cvtColor(save_img_np, cv2.COLOR_RGB2BGR)
        ldmk_img = draw_ldmk(save_img_cv2, landmarks[i], current_step)
        ldmk_img_pil = Image.fromarray(ldmk_img)
        ldmk_img_tensor = transforms.ToTensor()(ldmk_img_pil)
        ldmk_img_tensor = ldmk_img_tensor.unsqueeze(0)
        stack_img = torch.cat((stack_img.cuda(), ldmk_img_tensor.cuda()))
    make_grid_all_images = utils.make_grid(stack_img, 5)
    utils.save_image(make_grid_all_images, save_dir + 'gt_ldmk_' + str(current_step) + ".jpg")
    # import pdb;pdb.set_trace()