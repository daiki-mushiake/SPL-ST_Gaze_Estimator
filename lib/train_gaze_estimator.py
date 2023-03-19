'''
	This file includes the class to train SimGAN
	
	TrainSimGAN inherits the functions from 
	SubSimGAN. 

	SubSimGAN has the functions for 
	weight loading, initializing 
	data loaders, and accuracy metrics
'''

from locale import currency
import os

import torch
import torch.nn as nn
import torchvision
import cv2
import tensorboardX as tbx
import numpy as np
import argparse
import torch.nn.functional as F
import torch.autograd as autograd
import copy
import yaml
from torchvision import transforms,utils, models
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, median_filter
import math
import tqdm
import csv
import random
import torch.utils.data as Data
from sub_gaze_estimator import GazeEstimator
from torchvision.utils import save_image
from tqdm import tqdm
from torch.autograd import Variable
from util.calculate_optical_flow import optical_flow_calc, after_optical_flow_calc, before_optical_flow_calc ,save_refine_img
from util.calculate_gradient_graph import grad_feature_loss
from util.eval_images import eval_synthe_gazemap, eval_synthe_img, eval_real_img_landmark, eval_synthe_img_landmark, confirm_ldmk
from core.mask_generator import mask_generator
from util.gazemap_tensor import gazemap_generator
from util.data_augmentator import data_augmentation
from util.calibration_gaze import calibrated_gaze
from losses.losses import HeatmapLoss, GazeLoss, LandmarkLoss
from evaluation.evaluation_gaze import EvaluationGazeEstimator

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)

fake_num = len(os.listdir(config['synthetic_image_directory']['dirname']))
real_num = len(os.listdir(config['real_image_directory']['dirname']))


name_num = 1
sum_mean = 0 
sum_variance = 0

class TrainGazeEstimator(GazeEstimator):
    def __init__(self):
        GazeEstimator.__init__(self)
        self.writer = tbx.SummaryWriter(config['log_directory']['dirname'])
        self.recon_loss = None
        self.refiner_loss = None
        self.g_global_refined_adv_loss = None
        self.g_local_refined_adv_loss = None
        self.g_global_real_adv_loss = None
        self.g_local_real_adv_loss = None
        self.global_adv_loss = None
        self.local_adv_loss = None
        self.loss_real = None
        self.loss_refined = None
        self.refined_gaze_loss = None
        self.classifier_cross_entropy_loss = None
        self.edge_recon_loss = None
        self.grad_loss = None
        self.deformation_loss = None
        self.refined_gazemap_loss = None
        self.synthetic_gazemap_loss = None
        self.pred_refined_gaze_loss_mean = None
        self.pred_synthetic_gaze_loss_mean = None        

        self.mpiigaze_valid = 0
        self.utmultiview_valid = 0

        self.nstack = 3
        self.c = 0.01
        self.train_step = 0
        self.pretrain_step = 0
        self.refiner_2_pretrain_step = 0
        self.output_cnt = 0
        self.up_scaler = nn.Upsample(scale_factor=11, mode='bilinear') 
        self.D1_global_weight = 5
        self.D1_local_weight = 5
        self.D2_global_weight = 1
        self.D2_local_weight = 1
        self.edge_recon_weight = 1
        self.gaze_loss = GazeLoss()
        self.calc_losses = LandmarkLoss()

    def update_refiner(self, pretrain=False):
        '''Refiner1'''
        ''' Get batch of synthetic images '''
        synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = next(self.synthetic_data_iter)
        synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = data_augmentation(synthetic_images, gt_landmarks, gt_gaze, self.current_step)

        synthetic_images = synthetic_images.cuda()
        gt_heatmaps = gt_heatmaps.cuda()
        gt_landmarks = gt_landmarks.cuda()
        gt_gaze = gt_gaze.cuda()


        edge_synthetic_images = self.edge_detection(synthetic_images)


        '''Confirm GT landmarks'''
        # confirm_ldmk()

        ''' Get batch of Real images '''
        real_images = next(self.real_data_iter)
        real_images = real_images.cuda()
        edge_real_images = self.edge_detection(real_images)
        
        '''Refine Synthetic → Real images '''
        cg_to_real_images = self.R1(synthetic_images)
        cg_to_real_images = torch.sigmoid(cg_to_real_images)
        edge_cg_to_real_images = self.edge_detection(cg_to_real_images)
        
        '''Return Synthetic ← Real images'''
        return_real_to_cg_images = self.R2(cg_to_real_images)
        return_real_to_cg_images = torch.sigmoid(return_real_to_cg_images)
        edge_return_real_to_cg_images = self.edge_detection(return_real_to_cg_images)
        self.cycle_fake_recon_loss = self.feature_loss(synthetic_images, return_real_to_cg_images) + self.feature_loss(edge_synthetic_images, edge_return_real_to_cg_images)

        """Edge Reconstruction Loss"""
        mask = mask_generator(edge_synthetic_images, gt_landmarks[:,0:15,:], gt_landmarks[:,16:31,:])
        masked_edge_synthetic_images = mask * edge_synthetic_images
        masked_edge_cg_to_real_images = mask * edge_cg_to_real_images
        # masked_edge_images = torch.cat((masked_edge_synthetic_images ,masked_edge_cg_to_real_images),dim = 0)
        # utils.save_image(masked_edge_images, config['save_train_image_directory']['dirname'] + 'masked_img_' + str(self.current_step) + '.png')
        self.edge_recon_loss = self.feature_loss(masked_edge_synthetic_images ,masked_edge_cg_to_real_images) * self.edge_recon_weight

        """Optical Flow Synthetic ⇔ Real"""
        gt_landmarks_raft = gt_landmarks.clone()
        # self.deformation_loss, _ = optical_flow_calc(synthetic_images, cg_to_real_images, self.RAFT, gt_landmarks_raft)
        self.deformation_loss, _ = after_optical_flow_calc(synthetic_images, cg_to_real_images, self.RAFT, gt_landmarks_raft)
        # self.deformation_loss, _ = before_optical_flow_calc(synthetic_images, cg_to_real_images, self.RAFT, gt_landmarks_raft)

        '''Calculate Gradient Loss'''
        # self.grad_loss = grad_feature_loss(self.current_step, synthetic_images, cg_to_real_images)
        
        """refiner 2"""
        '''Real → Synthetic images'''
        real_to_cg_images = self.R2(real_images)
        real_to_cg_images = torch.sigmoid(real_to_cg_images)
        edge_real_to_cg_images = self.edge_detection(real_to_cg_images)

        '''Return Synthe → Real'''
        return_cg_to_real_images = self.R1(real_to_cg_images)
        return_cg_to_real_images = torch.sigmoid(return_cg_to_real_images)
        edge_return_cg_to_real_images = self.edge_detection(return_cg_to_real_images)
        # self.recon_loss2 = (self.feature_loss(refined_to_fake_images, real_images) + self.feature_loss(edge_refined_to_fake_images, edge_real_images))/2
        self.cycle_real_recon_loss = self.feature_loss(real_images, return_cg_to_real_images) + self.feature_loss(edge_real_images, edge_return_cg_to_real_images)


        ''' Get Discriminators predictions
            on the refined images '''

        '''Calculate G_D1 & L_D1 Adv Loss'''
        cat_refined_data = torch.cat((cg_to_real_images, edge_cg_to_real_images.cuda()), dim=1)
        # g_global_refined_predictions = self.G_D1(cat_refined_data)
        g_local_refined_predictions = self.L_D1(cat_refined_data)
        # self.g_global_refined_adv_loss = -torch.mean(g_global_refined_predictions) * self.D1_global_weight
        self.g_local_refined_adv_loss = -torch.mean(g_local_refined_predictions) * self.D1_local_weight
        
        '''Calculate G_D2 & L_D2 Adv Loss'''
        cat_real_data = torch.cat((real_to_cg_images, edge_real_to_cg_images.cuda()), dim=1)
        # g_global_real_predictions = self.G_D2(cat_real_data)
        g_local_real_predictions = self.L_D2(cat_real_data)
        # self.g_global_real_adv_loss = -torch.mean(g_global_real_predictions) * self.D2_global_weight
        self.g_local_real_adv_loss = -torch.mean(g_local_real_predictions) * self.D2_local_weight

        identity_loss = self.feature_loss(real_images, real_to_cg_images)

        if not pretrain:
            
            '''Calculate Gaze Loss'''
            refined_heatmaps_pred, refined_landmarks_pred = self.Reg(cg_to_real_images)
            self.refined_heatmap_loss, self.refined_landmarks_loss = self.calc_losses(refined_heatmaps_pred, refined_landmarks_pred, gt_heatmaps, gt_landmarks)
            # pred_refined_gazemap = gazemap_generator(cg_to_real_images, refined_landmarks_pred)
            refined_gaze = self.Densenet(cg_to_real_images)        
            refined_gaze_loss = self.gaze_loss(gt_gaze, refined_gaze)
            print('refined_gaze_loss:',refined_gaze_loss)
            self.refined_gaze_loss = refined_gaze_loss.mean() * 1000
            
            '''Calculate Heatmap Loss '''
            self.refined_heatmap_loss = self.refined_heatmap_loss.mean() * 1000

            '''Correct Loss'''
            #SP-ST GAN
            shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.refined_gaze_loss + self.deformation_loss + self.edge_recon_loss + identity_loss #Method1
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.deformation_loss + self.edge_recon_loss + identity_loss #method_2
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.deformation_loss + identity_loss #Method_3
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.deformation_loss + identity_loss #method_4
            # shape_preserving_loss = self.refined_heatmap_loss + self.refined_landmarks_loss + self.deformation_loss # Method_5
            # shape_preserving_loss = self.refined_gaze_loss + self.deformation_loss #Method_6
            # shape_preserving_loss = self.deformation_loss + self.edge_recon_loss #Method_7
            # shape_preserving_loss = self.deformation_loss #Method_8
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.refined_gaze_loss + self.deformation_loss + identity_loss #Method_9
            # shape_preserving_loss = self.refined_gaze_loss + self.deformation_loss + self.edge_recon_loss #Method_10
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_gaze_loss + self.deformation_loss + self.edge_recon_loss + identity_loss #Method_11
            # shape_preserving_loss = self.refined_heatmap_loss + self.refined_landmarks_loss + self.refined_gaze_loss + self.deformation_loss + self.edge_recon_loss + identity_loss #Method_12
            # shape_preserving_loss = self.refined_gaze_loss + self.edge_recon_loss #Method_13
            # shape_preserving_loss = self.edge_recon_loss #Method14
            # shape_preserving_loss = self.refined_gaze_loss #Method15
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.edge_recon_loss + self.refined_gaze_loss + identity_loss #method_16
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.edge_recon_loss + identity_loss #method_17
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + self.refined_heatmap_loss + self.refined_landmarks_loss + self.refined_gaze_loss + identity_loss #method_18
            # shape_preserving_loss = self.refined_heatmap_loss + self.refined_landmarks_loss # Method_19


            #SimGAN
            # identity_loss_A = self.feature_loss(cg_to_real_images, synthetic_images)*self.identity_A_weight
            # shape_preserving_loss = identity_loss + self.g_local_refined_adv_loss

            #CycleGAN
            # identity_loss_A = self.feature_loss(cg_to_real_images, synthetic_images)*self.identity_A_weight
            # identity_loss_B = self.feature_loss(real_to_cg_images, real_images)*self.identity_B_weight
            # identity_loss = identity_loss_A + identity_loss_B
            # shape_preserving_loss = self.cycle_fake_recon_loss + self.cycle_real_recon_loss + identity_loss + self.g_local_refined_adv_loss + self.g_local_real_adv_loss

            print('shape_preserving_loss: ',shape_preserving_loss)
            
            '''Losses Backward Part'''
            self.refiner1_optimizer.zero_grad()
            self.refiner2_optimizer.zero_grad()
            
            '''Adv Loss Backward'''
            # self.g_global_refined_adv_loss.backward(retain_graph=True)
            self.g_local_refined_adv_loss.backward(retain_graph=True)
            # self.g_global_real_adv_loss.backward(retain_graph=True)
            self.g_local_real_adv_loss.backward(retain_graph=True)
            print('local_refined_adv_loss: ',self.g_local_refined_adv_loss)
            # print('local_real_adv_loss:', self.g_local_real_adv_loss)

            '''Correct Loss Backward'''
            shape_preserving_loss.backward(retain_graph=True)

            self.refiner1_optimizer.step()	
            self.refiner2_optimizer.step()

        else:
            l1_loss_A = self.feature_loss(cg_to_real_images, synthetic_images)
            l1_loss_B = self.feature_loss(return_real_to_cg_images, synthetic_images)
            l1_loss_C = self.feature_loss(real_to_cg_images, real_images)
            l1_loss_D = self.feature_loss(return_cg_to_real_images, real_images)

            '''Losses Backward Part'''
            self.refiner1_optimizer.zero_grad()
            self.refiner2_optimizer.zero_grad()

            '''Correct Loss Backward'''
            shape_preserving_loss = l1_loss_A + l1_loss_B + l1_loss_C + l1_loss_D
            shape_preserving_loss.backward(retain_graph=True)

            self.refiner1_optimizer.step()
            self.refiner2_optimizer.step()

            # if self.pretrain_step % 100 == 0:
            #     save_dir = '../refine_img/'
            #     utils.save_image(cg_to_real_images , save_dir + 'pretrain_cg_to_real_images_' + str(self.pretrain_step) + ".jpg")
            #     utils.save_image(return_real_to_cg_images , save_dir + 'pretrain_return_real_to_cg_images_' + str(self.pretrain_step) + ".jpg")
            #     utils.save_image(real_to_cg_images , save_dir + 'pretrain_real_to_cg_images_' + str(self.pretrain_step) + ".jpg")
            #     utils.save_image(return_cg_to_real_images , save_dir + 'pretrain_return_cg_to_real_images_' + str(self.pretrain_step) + ".jpg")
            # self.pretrain_step += 1     

    def pretrain_refiner(self):
        # This method pretrains the generator if called
        print('Pre-training the refiner network {} times'.format(config['refiner_pretrain_iteration']['num']))
        ''' Set the refiner gradients parameters to True 
            Set the discriminators gradients params to False'''

        self.R1.train()
        for param in self.R1.parameters():
            param.requires_grad = True

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.train()		
        for param in self.R2.parameters():
            param.requires_grad = True
        
        self.G_D1.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        
        self.G_D2.eval()
        for param in self.G_D2.parameters():
            param.requires_grad = False
        
        self.L_D1.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False

        self.L_D2.eval()
        for param in self.L_D2.parameters():
            param.requires_grad = False

        self.Reg.eval()
        for param in self.Reg.parameters():
            param.requires_grad = False  

        # self.Hourglass_net.eval()
        # for param in self.Hourglass_net.parameters():
        #     param.requires_grad = False

        self.Densenet.eval()
        for param in self.Densenet.parameters():
            param.requires_grad = False
        
        ''' Begin pre-training the refiner '''
        for step in tqdm(range(config['refiner_pretrain_iteration']['num'])):
            self.update_refiner(pretrain=True)

    def train_refiner(self):
        self.R1.train()
        for param in self.R1.parameters():
            param.requires_grad = True

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.train()		
        for param in self.R2.parameters():
            param.requires_grad = True
        
        self.G_D1.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        
        self.G_D2.eval()
        for param in self.G_D2.parameters():
            param.requires_grad = False
        
        self.L_D1.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False

        self.L_D2.eval()
        for param in self.L_D2.parameters():
            param.requires_grad = False

        self.Reg.eval()
        for param in self.Reg.parameters():
            param.requires_grad = False  

        # self.Hourglass_net.eval()
        # for param in self.Hourglass_net.parameters():
        #     param.requires_grad = False

        self.Densenet.eval()
        for param in self.Densenet.parameters():
            param.requires_grad = False

        for idx in range(config['updates_refiner_per_step']['num']):
            self.update_refiner(pretrain=False)

    #-----------------------------------------------------Discriminator---------------------------------------------------------------------------#
    def update_discriminator(self, pretrain=False):
    
        ''' get batch of real images '''        
        real_images = next(self.real_data_iter)
        real_images = real_images.cuda()
        real_images = torch.cat((real_images, real_images), dim=0)

        '''Create Real Edge Image'''
        edge_real_images = self.edge_detection(real_images)
        cat_real = torch.cat((real_images, edge_real_images.cuda()), dim=1).cuda()

        ''' get batch of synthetic images '''
        synthetic_images, _, gt_landmarks, gt_gaze = next(self.synthetic_data_iter)
        synthetic_images, _, _, _ = data_augmentation(synthetic_images, gt_landmarks, gt_gaze, self.current_step)
        synthetic_images = synthetic_images.cuda()


        '''Create Synthetic Edge Image'''
        edge_synthetic_images = self.edge_detection(torch.cat((synthetic_images, synthetic_images),dim=0))
        cat_synthetic = torch.cat((torch.cat((synthetic_images, synthetic_images), dim=0), edge_synthetic_images.cuda()), dim=1)

        if not pretrain:
            cg_to_real_images_R1_past = self.R1_past(synthetic_images)
            cg_to_real_images_R1 = self.R1(synthetic_images)
            cg_to_real_images = torch.cat((cg_to_real_images_R1_past, cg_to_real_images_R1), dim=0)
            cg_to_real_images = torch.sigmoid(cg_to_real_images)
            
            '''Create synthetic Edge Image'''
            edge_cg_to_real_images = self.edge_detection(cg_to_real_images)
            cat_refined = torch.cat((cg_to_real_images, edge_cg_to_real_images.cuda()), dim=1).cuda()


            #---------------Global Discriminator 1-------------------------------------------
            '''Calculate & Backward G_D1 Adv Loss'''
            global_real_predictions = self.G_D1(cat_real).cuda()
            global_refined_predictions = self.G_D1(cat_refined).cuda()
            # global_real_predictions = self.G_D1(real_images).cuda()  
            # global_refined_predictions = self.G_D1(cg_to_real_images).cuda()

            '''Wassernstein Loss'''
            train_global_gradient_penalty = self.global_compute_gradient_penalty(self.G_D1, cat_real, cat_refined).cuda()     
            self.global_discriminator1_loss = -torch.mean(global_real_predictions) + torch.mean(global_refined_predictions) + (10 * train_global_gradient_penalty)

            '''MSE Loss'''
            # global_real_predictions = self.adv_loss(global_real_predictions, torch.tensor(1.0).expand_as(global_real_predictions).cuda())
            # global_refined_predictions = self.adv_loss(global_refined_predictions, torch.tensor(0.0).expand_as(global_refined_predictions).cuda())
            # self.global_discriminator1_loss = (( global_real_predictions + global_refined_predictions) * 0.5)+ (10 * train_global_gradient_penalty)

            '''Global Discriminator1 Back Ward'''
            self.global_discriminator1_optimizer.zero_grad()
            self.global_discriminator1_loss.backward(retain_graph=True)
            self.global_discriminator1_optimizer.step()


            #---------------Local Discriminator 1-------------------------------------------
            '''Calculate & Backward L_D1 Adv Loss'''
            local_refined_predictions = self.L_D1(cat_refined).cuda()
            local_real_predictions = self.L_D1(cat_real).cuda()
            # local_refined_predictions = self.L_D1(cg_to_real_images).cuda()
            # local_real_predictions = self.L_D1(real_images).cuda()

            '''Wassernstein Loss'''
            train_local_gradient_penalty = self.local_compute_gradient_penalty(self.L_D1, cat_real, cat_refined).cuda()
            self.local_discriminator1_loss = -torch.mean(local_real_predictions) + torch.mean(local_refined_predictions) + (10 * train_local_gradient_penalty)

            '''MSE Loss'''
            # local_refined_predictions = self.adv_loss(local_refined_predictions, torch.tensor(0.0).expand_as(local_refined_predictions).cuda())
            # local_real_predictions  = self.adv_loss(local_real_predictions, torch.tensor(1.0).expand_as(local_real_predictions).cuda())
            # self.local_discriminator1_loss = ((local_refined_predictions + local_real_predictions) * 0.5) + (10 * train_local_gradient_penalty)

            '''Local Discriminator1 Back Ward'''
            self.local_discriminator1_optimizer.zero_grad()
            self.local_discriminator1_loss.backward(retain_graph=True)
            self.local_discriminator1_optimizer.step()
        else:
            #---------------Global Discriminator 1-------------------------------------------
            '''Calculate & Backward G_D1 Adv Loss'''
            global_synthetic_predictions = self.G_D1(cat_synthetic).cuda()
            global_real_predictions = self.G_D1(cat_real).cuda()
            # global_synthetic_predictions = self.G_D1(synthetic_images).cuda()
            # global_real_predictions = self.G_D1(real_images).cuda()

            '''Wassernstein Loss'''
            pretrain_global_gradient_penalty = self.global_compute_gradient_penalty(self.G_D1, cat_real, cat_synthetic).cuda()
            self.global_discriminator1_loss = -torch.mean(global_real_predictions) + torch.mean(global_synthetic_predictions) + (10 * pretrain_global_gradient_penalty)

            '''MSE Loss'''
            # global_synthetic_predictions = self.adv_loss(global_synthetic_predictions, torch.tensor(0.0).expand_as(global_synthetic_predictions).cuda())
            # global_real_predictions = self.adv_loss(global_real_predictions, torch.tensor(1.0).expand_as(global_real_predictions).cuda())
            # self.global_discriminator1_loss = (( global_real_predictions + global_synthetic_predictions) * 0.5) + (10 * pretrain_global_gradient_penalty)

            '''Global Discriminator1 Back Ward'''
            self.global_discriminator1_optimizer.zero_grad()
            self.global_discriminator1_loss.backward(retain_graph=True)
            self.global_discriminator1_optimizer.step()

            #---------------Local Discriminator 1-------------------------------------------
            '''Calculate & Backward L_D1 Adv Loss'''
            local_real_predictions = self.L_D1(cat_real).cuda()
            local_synthetic_predictions = self.L_D1(cat_synthetic).cuda()

            '''Wassernstein Loss'''
            pretrain_local_gradient_penalty = self.local_compute_gradient_penalty(self.L_D1, cat_real, cat_synthetic).cuda()
            self.local_discriminator1_loss = -torch.mean(local_real_predictions) + torch.mean(local_synthetic_predictions) + (10 * pretrain_local_gradient_penalty)

            '''MSE Loss'''
            # local_real_predictions  = self.adv_loss(local_real_predictions, torch.tensor(1.0).expand_as(local_real_predictions).cuda())
            # local_synthetic_predictions = self.adv_loss(local_synthetic_predictions, torch.tensor(0.0).expand_as(local_synthetic_predictions).cuda())
            # self.local_discriminator1_loss = (local_real_predictions + local_synthetic_predictions) * 0.5

            '''Local Discriminator1 Back Ward'''
            self.local_discriminator1_optimizer.zero_grad()
            self.local_discriminator1_loss.backward(retain_graph=True)
            self.local_discriminator1_optimizer.step()

        if not pretrain:
            '''Calculate & Backward G_D2 Adv Loss'''
            real_to_cg_images = self.R2(real_images)
            real_to_cg_images = torch.sigmoid(real_to_cg_images)
            edge_real_to_cg_images = self.edge_detection(real_to_cg_images)
            cat_real_to_cg = torch.cat((real_to_cg_images, edge_real_to_cg_images), dim=1).cuda()

            #---------------Global Discriminator 2-------------------------------------------
            global_synthetic_predictions2 = self.G_D2(cat_synthetic).cuda()
            global_fake_predictions2 = self.G_D2(cat_real_to_cg).cuda()
            # global_synthetic_predictions2 = self.G_D2(synthetic_images).cuda()        
            # global_fake_predictions2 = self.G_D2(real_to_cg_images).cuda()

            '''Wassernstein Loss'''
            train_global_gradient_penalty2 = self.global_compute_gradient_penalty(self.G_D2, cat_synthetic, cat_real_to_cg).cuda()
            self.global_discriminator2_loss = -torch.mean(global_synthetic_predictions2) + torch.mean(global_fake_predictions2) + (10 * train_global_gradient_penalty2)

            '''MSE Loss'''
            # global_synthetic_predictions2 = self.adv_loss(global_synthetic_predictions2, torch.tensor(1.0).expand_as(global_synthetic_predictions2).cuda())
            # global_fake_predictions2 = self.adv_loss(global_fake_predictions2, torch.tensor(0.0).expand_as(global_fake_predictions2).cuda())
            # self.global_discriminator2_loss = ((global_synthetic_predictions2 + global_fake_predictions2) * 0.5) + (10 * train_global_gradient_penalty2)

            '''Global Discriminator2 Back Ward'''
            self.global_discriminator2_optimizer.zero_grad()
            self.global_discriminator2_loss.backward(retain_graph=True)
            self.global_discriminator2_optimizer.step()

            #---------------Local Discriminator 2-------------------------------------------
            '''Calculate & Backward L_D2 Adv Loss'''
            real_to_cg_images = self.R2(real_images)
            real_to_cg_images = torch.sigmoid(real_to_cg_images)
            edge_real_to_cg_images = self.edge_detection(real_to_cg_images)
            cat_real_to_cg = torch.cat((real_to_cg_images, edge_real_to_cg_images), dim=1).cuda()
            local_fake_predictions2 = self.L_D2(cat_real_to_cg).cuda()
            local_synthetic_predictions2 = self.L_D2(cat_synthetic).cuda()

            '''Wassernstein Loss'''
            train_local_gradient_penalty2 = self.local_compute_gradient_penalty(self.L_D2, cat_synthetic, cat_real_to_cg).cuda()
            self.local_discriminator2_loss = -torch.mean(local_synthetic_predictions2) + torch.mean(local_fake_predictions2) + (10 * train_local_gradient_penalty2)

            '''MSE Loss'''
            # local_fake_predictions2 = self.adv_loss(local_fake_predictions2, torch.tensor(0.0).expand_as(local_fake_predictions2).cuda())
            # local_synthetic_predictions2  = self.adv_loss(local_synthetic_predictions2, torch.tensor(1.0).expand_as(local_synthetic_predictions2).cuda())
            # self.local_discriminator2_loss = ((local_fake_predictions2 + local_synthetic_predictions2) * 0.5) + (10 * train_local_gradient_penalty2)

            '''Local Discriminator2 Back Ward'''
            self.local_discriminator2_optimizer.zero_grad()
            self.local_discriminator2_loss.backward(retain_graph=True)
            self.local_discriminator2_optimizer.step()
        else:
            #---------------Global Discriminator 2-------------------------------------------            
            '''Calculate & Backward G_D2 Adv Loss'''
            global_synthetic_predictions2 = self.G_D2(cat_synthetic).cuda()
            global_real_predictions2 = self.G_D2(cat_real).cuda()
            # global_synthetic_predictions2 = self.G_D2(synthetic_images).cuda()
            # global_real_predictions2 = self.G_D2(real_images).cuda()

            '''Wassernstein Loss'''
            pretrain_global_gradient_penalty2 = self.global_compute_gradient_penalty(self.G_D2, cat_synthetic, cat_real).cuda()
            self.global_discriminator2_loss = -torch.mean(global_synthetic_predictions2) + torch.mean(global_real_predictions2) + (10 * pretrain_global_gradient_penalty2)

            '''MSE Loss'''
            # global_synthetic_predictions2 = self.adv_loss(global_synthetic_predictions2, torch.tensor(1.0).expand_as(global_synthetic_predictions2).cuda())
            # global_real_predictions2 = self.adv_loss(global_real_predictions2, torch.tensor(0.0).expand_as(global_real_predictions2).cuda())
            # self.global_discriminator2_loss = (global_synthetic_predictions2 + global_real_predictions2) * 0.5

            '''Global Discriminator2 Back Ward'''
            self.global_discriminator2_optimizer.zero_grad()
            self.global_discriminator2_loss.backward(retain_graph=True)
            self.global_discriminator2_optimizer.step()

            #---------------Local Discriminator 2-------------------------------------------
            '''Calculate & Backward L_D2 Adv Loss'''
            local_synthetic_predictions2 = self.L_D2(cat_synthetic).cuda()
            local_real_predictions2 = self.L_D2(cat_real).cuda()   
            # local_synthetic_predictions2 = self.L_D2(synthetic_images).cuda()
            # local_real_predictions2 = self.L_D2(real_images).cuda()   

            '''Wassernstein Loss'''
            pretrain_local_gradient_penalty2 = self.local_compute_gradient_penalty(self.L_D2, cat_synthetic, cat_real).cuda()
            self.local_discriminator2_loss = -torch.mean(local_synthetic_predictions2) + torch.mean(local_real_predictions2) + (10 * pretrain_local_gradient_penalty2)

            '''MSE Loss'''
            # local_synthetic_predictions2  = self.adv_loss(local_synthetic_predictions2, torch.tensor(1.0).expand_as(local_synthetic_predictions2).cuda())
            # local_real_predictions2 = self.adv_loss(local_real_predictions2, torch.tensor(0.0).expand_as(local_real_predictions2).cuda())    
            # self.local_discriminator2_loss = (local_real_predictions2 + local_synthetic_predictions2) * 0.5

            '''Local Discriminator2 Back Ward'''
            self.local_discriminator2_optimizer.zero_grad()
            self.local_discriminator2_loss.backward(retain_graph=True)
            self.local_discriminator2_optimizer.step()

    def pretrain_discriminator(self):
        print('Pre-training the discriminator network {} times'.format(config['discriminator_pretrain_iteration']['num']))

        ''' Set the Discriminators gradient parameters to True
            Set the Refiners gradient parameters to False '''

        self.R1.eval()
        for param in self.R1.parameters():
            param.requires_grad = False

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.eval()		
        for param in self.R2.parameters():
            param.requires_grad = False
        
        self.G_D1.train()
        for param in self.G_D1.parameters():
            param.requires_grad = True
        
        self.G_D2.train()
        for param in self.G_D2.parameters():
            param.requires_grad = True
        
        self.L_D1.train()
        for param in self.L_D1.parameters():
            param.requires_grad = True

        self.L_D2.train()
        for param in self.L_D2.parameters():
            param.requires_grad = True

        self.Reg.eval()
        for param in self.Reg.parameters():
            param.requires_grad = False

        # self.Hourglass_net.eval()
        # for param in self.Hourglass_net.parameters():
        #     param.requires_grad = False

        self.Densenet.eval()
        for param in self.Densenet.parameters():
            param.requires_grad = False

        ''' Begin pretraining the discriminator '''
        for step in tqdm(range(config['discriminator_pretrain_iteration']['num'])):
            ''' update discriminator and return some important info for printing '''
            self.update_discriminator(pretrain=True)

    def train_discriminator(self):
        '''Train Discriminator'''
        self.R1.eval()
        for param in self.R1.parameters():
            param.requires_grad = False

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.eval()		
        for param in self.R2.parameters():
            param.requires_grad = False
        
        self.G_D1.train()
        for param in self.G_D1.parameters():
            param.requires_grad = True
        
        self.G_D2.train()
        for param in self.G_D2.parameters():
            param.requires_grad = True
        
        self.L_D1.train()
        for param in self.L_D1.parameters():
            param.requires_grad = True

        self.L_D2.train()
        for param in self.L_D2.parameters():
            param.requires_grad = True

        self.Reg.eval()
        for param in self.Reg.parameters():
            param.requires_grad = False  

        # self.Hourglass_net.eval()
        # for param in self.Hourglass_net.parameters():
        #     param.requires_grad = False

        self.Densenet.eval()
        for param in self.Densenet.parameters():
            param.requires_grad = False
                
        for idx in range(config['updates_discriminator_per_step']['num']):
            self.update_discriminator(pretrain=False)


    #-----------------------------------------------------Regressor---------------------------------------------------------------------------#
    def update_regressor(self, pretrain=False):
        ''' Get batch of synthetic images '''
        # synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze, _ = next(self.synthetic_regressor_data_iter)
        synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = next(self.synthetic_regressor_data_iter)
        synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = data_augmentation(synthetic_images, gt_landmarks, gt_gaze, self.current_step)
        synthetic_images = synthetic_images.cuda()
        gt_heatmaps = gt_heatmaps.cuda()
        gt_landmarks = gt_landmarks.cuda()
        gt_gaze = gt_gaze.cuda()

        if not pretrain:
            ''' Refine synthetic images '''
            refined_images = self.R1(synthetic_images)
            refined_images = torch.sigmoid(refined_images)
            # self.refined_images, self.gt_heatmaps, self.gt_landmarks, _ = data_augmentation(self.refined_images, self.gt_landmarks, self.gt_gaze)		
            reg_refined_heatmaps_pred, reg_refined_landmarks_pred = self.Reg(refined_images.cuda())     
            
            """refined image heatmaploss & landmarkloss"""
            reg_refined_heatmap_loss, reg_refined_landmarks_loss = self.calc_losses(reg_refined_heatmaps_pred, reg_refined_landmarks_pred, gt_heatmaps, gt_landmarks)
            # self.diff_refine_ldmks = self.calc_landmark_diff(self.reg_refined_landmarks_pred, self.gt_landmarks)            
            reg_refined_heatmap_loss = reg_refined_heatmap_loss * 1000
            reg_return_refined_heatmap_loss = reg_refined_heatmap_loss.mean()
            reg_return_refined_landmark_loss = reg_refined_landmarks_loss.mean()
            
            """synthetic image heatmaploss & landmarkloss"""
            # self.reg_synthetic_heatmaps_pred, self.reg_synthetic_landmarks_pred = self.Reg(self.synthetic_images)
            # self.reg_synthetic_heatmap_loss, self.reg_synthetic_landmarks_loss = self.calc_losses(self.reg_synthetic_heatmaps_pred, self.reg_synthetic_landmarks_pred, self.gt_heatmaps, self.gt_landmarks)
            # self.synthe_ldmks_diff = self.calc_landmark_diff(self.reg_synthetic_landmarks_pred.cuda(), gt_landmarks.cuda())    
            # self.synthe_refine_ldmks_diff = self.calc_landmark_diff(self.reg_refined_landmarks_pred.cuda(), self.reg_synthetic_landmarks_pred.cuda())
            # self.synthetic_heatmap_loss2 = self.reg_synthetic_heatmap_loss * 1000
            # self.reg_return_synthetic_heatmap_loss = self.synthetic_heatmap_loss2.mean()
            # self.reg_return_synthetic_landmark_loss = self.reg_synthetic_landmarks_loss.mean()

            regressor_loss = reg_return_refined_heatmap_loss + reg_return_refined_landmark_loss
            # regressor_loss = self.reg_return_refined_heatmap_loss + self.reg_return_refined_landmark_loss + self.reg_return_synthetic_heatmap_loss + self.reg_return_synthetic_landmark_loss

            # self.reg_return_synthetic_heatmap_loss.backward(retain_graph=True)
            # self.reg_return_synthetic_landmark_loss.backward(retain_graph=True)

            self.regressor_optimizer.zero_grad()   
            regressor_loss.backward(retain_graph=True)
            self.regressor_optimizer.step()
        else:
            self.regressor_optimizer.zero_grad()
            self.pre_reg_synthetic_heatmaps_pred, self.pre_reg_synthetic_landmarks_pred = self.Reg(synthetic_images)

            """refined image heatmaploss & landmarkloss"""
            self.pre_reg_synthetic_heatmap_loss, self.pre_reg_synthetic_landmarks_loss = self.calc_losses(self.pre_reg_synthetic_heatmaps_pred, self.pre_reg_synthetic_landmarks_pred, gt_heatmaps, gt_landmarks)
            self.pre_synthetic_heatmap_loss2 = self.pre_reg_synthetic_heatmap_loss * 1000
            self.pre_return_heatmap_loss = self.pre_synthetic_heatmap_loss2.mean()
            self.pre_return_landmark_loss = self.pre_reg_synthetic_landmarks_loss.mean()
            # self.pre_return_heatmap_loss.backward(retain_graph=True)
            # self.pre_return_landmark_loss.backward(retain_graph=True)
            regressor_loss = self.pre_return_heatmap_loss + self.pre_return_landmark_loss
            regressor_loss.backward(retain_graph=True)
            self.regressor_optimizer.step()

    def pretrain_regressor(self):
        print('Pre-training the regressor network {} times'.format(config['regressor_pretrain_iteration']['num']))

        """Set the Classifiers gradient parameters to True"""
        """ Set the Refiners gradient parameters to False"""
        self.R1.eval()
        for param in self.R1.parameters():
            param.requires_grad = False

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.eval()		
        for param in self.R2.parameters():
            param.requires_grad = False
        
        self.G_D1.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        
        self.G_D2.eval()
        for param in self.G_D2.parameters():
            param.requires_grad = False
        
        self.L_D1.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False

        self.L_D2.eval()
        for param in self.L_D2.parameters():
            param.requires_grad = False

        self.Reg.train()
        for param in self.Reg.parameters():
            param.requires_grad = True  

        # self.Hourglass_net.eval()
        # for param in self.Hourglass_net.parameters():
        #     param.requires_grad = False

        self.Densenet.eval()
        for param in self.Densenet.parameters():
            param.requires_grad = False
        
        for step in tqdm(range(config['regressor_pretrain_iteration']['num'])):
            ''' update discriminator and return some important info for printing '''
            self.update_regressor(pretrain=True)

    def train_regressor(self):
        '''Train Regressor'''
        
        self.R1.eval()
        for param in self.R1.parameters():
            param.requires_grad = False

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.eval()		
        for param in self.R2.parameters():
            param.requires_grad = False
        
        self.G_D1.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        
        self.G_D2.eval()
        for param in self.G_D2.parameters():
            param.requires_grad = False
        
        self.L_D1.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False

        self.L_D2.eval()
        for param in self.L_D2.parameters():
            param.requires_grad = False

        self.Reg.train()
        for param in self.Reg.parameters():
            param.requires_grad = True  

        # self.Hourglass_net.eval()
        # for param in self.Hourglass_net.parameters():
        #     param.requires_grad = False

        self.Densenet.eval()
        for param in self.Densenet.parameters():
            param.requires_grad = False

        for idx in range(config['updates_regressor_per_step']['num']):
            self.update_regressor(pretrain=False)

    #-----------------------------------------------------Densenet---------------------------------------------------------------------------#
    def update_densenet(self, pretrain=False):
        '''Get batch of sythetic images'''
        synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = next(self.synthetic_regressor_data_iter)
        original_images = synthetic_images
        synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = data_augmentation(synthetic_images, gt_landmarks, gt_gaze, self.current_step)
        synthetic_images = synthetic_images.cuda()
        gt_heatmaps = gt_heatmaps.cuda()
        gt_landmarks = gt_landmarks.cuda()
        gt_gaze = gt_gaze.cuda()

        save_dir = '../augmentation_img/'
        if self.current_step % 100 == 0:
            utils.save_image(synthetic_images, save_dir + 'augmentation_' + str(self.current_step) + ".jpg")

        if not pretrain:
            '''Refine synthetic Images'''
            refined_images = self.R1(synthetic_images)
            refined_images = torch.sigmoid(refined_images)
            # refined_images = torch.clamp(refined_images, min=0, max=1)
        
            if self.current_step % 100 == 0:    
                save_dir = '../refine_img/'
                save_refine_img = torch.cat((original_images.cuda(), synthetic_images, refined_images),3)
                utils.save_image(save_refine_img , save_dir + 'Refine_' + str(self.current_step) + ".jpg")

            # steps = self.current_step
            # aug_images, aug_heatmaps, aug_landmarks, aug_gaze = data_augmentation(refined_images, gt_landmarks, gt_gaze, steps)

            # save_dir = '../augmentation_img/'
            # if self.current_step % 100 == 0:
            #     utils.save_image(aug_images, save_dir + 'augmentation_' + str(self.current_step) + ".jpg")

            # refined_images = torch.cat((refined_images, aug_images.cuda()), dim=0)
            # gt_landmarks = torch.cat((gt_landmarks, aug_landmarks.cuda()), dim=0)
            # gt_heatmaps = torch.cat((gt_heatmaps, aug_heatmaps.cuda()), dim=0)
            # gt_gaze = torch.cat((gt_gaze, aug_gaze.cuda()), dim=0)
            # _, pred_refined_gazemap = self.Hourglass_net(refined_images)
            pred_refined_gaze = self.Densenet(refined_images)
            pred_refined_gaze_loss = self.gaze_loss(gt_gaze, pred_refined_gaze)
            self.pred_refined_gaze_loss = pred_refined_gaze_loss.mean() * 1000

            # pred_gaze_loss = self.pred_refined_gaze_loss_mean + self.pred_synthetic_gaze_loss_mean
            pred_gaze_loss = self.pred_refined_gaze_loss
            self.Densenet_optimizer.zero_grad()
            pred_gaze_loss.backward()
            self.Densenet_optimizer.step()
        else:
            pred_synthetic_gaze = self.Densenet(synthetic_images)
            pred_synthetic_gaze_loss = self.gaze_loss(gt_gaze, pred_synthetic_gaze)
            self.pred_synthetic_gaze_loss_mean = pred_synthetic_gaze_loss.mean() * 1000

            self.Densenet_optimizer.zero_grad()
            self.pred_synthetic_gaze_loss_mean.backward()
            self.Densenet_optimizer.step()
        
        
    def pretrain_densenet(self):
        print('Pre-training the Densenet {} times'. format(config['densenet_pretrain_iteration']['num'] - 1))

        self.R1.eval()
        for param in self.R1.parameters():
            param.requires_grad = False

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.eval()		
        for param in self.R2.parameters():
            param.requires_grad = False
        
        self.G_D1.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        
        self.G_D2.eval()
        for param in self.G_D2.parameters():
            param.requires_grad = False
        
        self.L_D1.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False

        self.L_D2.eval()
        for param in self.L_D2.parameters():
            param.requires_grad = False

        self.Reg.eval()
        for param in self.Reg.parameters():
            param.requires_grad = False  

        self.Densenet.train()
        for param in self.Densenet.parameters():
            param.requires_grad = True

        for step in tqdm(range(config['densenet_pretrain_iteration']['num'])):
            self.update_densenet(pretrain=True)

    def train_densenet(self):

        self.R1.eval()
        for param in self.R1.parameters():
            param.requires_grad = False

        self.R1_past.eval()
        for param in self.R1_past.parameters():
            param.requires_grad = False

        self.R2.eval()		
        for param in self.R2.parameters():
            param.requires_grad = False
        
        self.G_D1.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        
        self.G_D2.eval()
        for param in self.G_D2.parameters():
            param.requires_grad = False
        
        self.L_D1.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False

        self.L_D2.eval()
        for param in self.L_D2.parameters():
            param.requires_grad = False

        self.Reg.eval()
        for param in self.Reg.parameters():
            param.requires_grad = False  

        self.Densenet.train()
        for param in self.Densenet.parameters():
            param.requires_grad = True

        for idx in range(config['updates_densenet_per_step']['num']):
            self.update_densenet(pretrain=False)
     
    def train(self):
        self.build_network()	
        self.get_data_loaders()

        """Load Past Refiner checkpoint"""
        self.load_R1_past_weights()

        if not self.weights_loaded:
            self.pretrain_refiner()
            self.pretrain_discriminator()
            self.pretrain_regressor()
            self.pretrain_densenet()

            torch.save(self.R1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_refiner1_path']['pathname']))
            torch.save(self.R2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_refiner2_path']['pathname']))
            torch.save(self.G_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_global_discriminator1_path']['pathname']))
            torch.save(self.L_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_local_discriminator1_path']['pathname']))
            torch.save(self.G_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_global_discriminator2_path']['pathname']))
            torch.save(self.L_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_local_discriminator2_path']['pathname']))
            torch.save(self.Reg.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_regressor_path']['pathname']))
            torch.save(self.Densenet.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_densenet_path']['pathname']))
            
            state = {
                    'step': 0,
                    'optG_D_1' : self.global_discriminator1_optimizer.state_dict(),
                    'optG_D_2' : self.global_discriminator2_optimizer.state_dict(),
                    'optL_D_1' : self.local_discriminator1_optimizer.state_dict(),
                    'optL_D_2' : self.local_discriminator2_optimizer.state_dict(),
                    'optR1' : self.refiner1_optimizer.state_dict(),
                    'optR2' : self.refiner2_optimizer.state_dict(),
                    'optReg' : self.regressor_optimizer.state_dict(),
                    'optDensenet' : self.Densenet_optimizer.state_dict()
                }
            print("save optimizer_path")
            torch.save(state, os.path.join(config['checkpoint_path']['pathname'], config['pretrain_optimizer_path']['pathname']))

        print("Train Start")
        assert self.current_step < config['train_iteration']['num'], 'Target step is smaller than current step'
        for step in tqdm(range((self.current_step + 1), config['train_iteration']['num'])):
            print(os.getcwd())
            print("Synthetic Image:",config['synthetic_image_directory']['dirname'])
            print("Train:",config['real_image_directory']['dirname'])
            print("Test:",config['real_image_test_directory']['dirname'])

            self.current_step = step

            """Load Past Refiner checkpoint"""
            self.load_R1_past_weights()

            '''Train Step'''
            self.train_refiner()
            self.train_discriminator()
            self.train_regressor()
            self.train_densenet()       
            
            if config['log']['bool'] == True and (step % config['log_interval']['num'] == 0 or step == 0):
                '''Eval Predict landmark'''
                # self.eval_real_img_landmark()
                # self.eval_synthe_img_landmark()
                # eval_real_img_landmark(self.Densenet, self.real_eval_data_iter, self.current_step)
                # eval_synthe_img_landmark(self.R1_past, self.Densenet, self.synthe_eval_data_iter, self.current_step)
                eval_synthe_img(self.R1, self.Reg, self.real_eval_data_iter, self.eval_data_iter, self.current_step)
                # eval_synthe_gazemap(self.Reg, self.synthe_eval_data_iter, self.current_step)

                '''Evaluation Gaze'''
                self.evaluation_gaze = EvaluationGazeEstimator(self.Densenet, self.current_step)
                self.mpiigaze_valid = self.evaluation_gaze.evaluation_gaze_estimator('mpiigaze_validation', self.mpii_validation_data_iter)
                self.mpiigaze_test = self.evaluation_gaze.evaluation_gaze_estimator('mpiigaze_test', self.mpii_test_data_iter)
                self.utmultiview_valid = self.evaluation_gaze.evaluation_gaze_estimator('ut_multiview_validation',self.ut_validation_data_iter)
                self.utmultiview_test = self.evaluation_gaze.evaluation_gaze_estimator('ut_multiview_test',self.ut_test_data_iter)
                self.columbia_valid = self.evaluation_gaze.evaluation_gaze_estimator('columbia_validation',self.columbia_validation_iter)
                self.columbia_test = self.evaluation_gaze.evaluation_gaze_estimator('columbia_test',self.columbia_test_iter)
                self.eth_valid = self.evaluation_gaze.evaluation_gaze_estimator('eth_xgaze_validation_left',self.eth_validation_data_iter)
                self.eth_test = self.evaluation_gaze.evaluation_gaze_estimator('eth_xgaze_test_left',self.eth_test_data_iter)

                # self.mpiigaze_calib_test = calibrated_gaze(self.Densenet, self.current_step)

                '''Write Tensorflow log data'''
                print("step = ", step)
                # self.writer.add_scalar("Refiner/cycle fake recon loss ", self.cycle_fake_recon_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/cycle real recon loss ", self.cycle_real_recon_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/Global refined adversarial loss ", self.g_global_refined_adv_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/Local refined adversarial loss ", self.g_local_refined_adv_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/Global real adversarial loss ", self.g_global_real_adv_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/Local real adversarial loss ", self.g_local_real_adv_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/deformation loss", self.deformation_loss.item(), global_step = step)
                # self.writer.add_scalar("Refiner/Edge recon loss", self.edge_recon_loss.item(), global_step = step)
                # self.writer.add_scalar("Discriminator/Global Discriminator1 Adversarial loss", self.global_discriminator1_loss.item(), global_step = step)
                # self.writer.add_scalar("Discriminator/Local Discriminator1 Adversarial loss", self.local_discriminator1_loss.item(), global_step = step)
                # self.writer.add_scalar("Discriminator/Global Discriminator2 Adversarial loss", self.global_discriminator2_loss.item(), global_step = step)
                # self.writer.add_scalar("Discriminator/Local Discriminator2 Adversarial loss", self.local_discriminator2_loss.item(), global_step = step)
                # self.writer.add_scalar("Regressor/refined heatmap loss", self.reg_return_refined_heatmap_loss.item(), global_step = step)
                # self.writer.add_scalar("Regressor/refined landmark loss", self.reg_return_refined_landmark_loss.item(), global_step = step)
                # self.writer.add_scalar("Regressor/synthetic heatmap loss", self.reg_return_synthetic_heatmap_loss.item(), global_step = step)
                # self.writer.add_scalar("Regressor/synthetic landmark loss", self.reg_return_synthetic_landmark_loss.item(), global_step = step)
                # self.writer.add_scalar("Gaze Estimator/refined gaze loss", self.reg_return_refined_gaze_loss.item(), global_step = step)
                self.writer.add_scalar("Gaze Estimator/utmultiview valid mean diff gaze", self.utmultiview_valid.item(), global_step = step)
                self.writer.add_scalar("Gaze Estimator/utmultiview test mean diff gaze", self.utmultiview_test.item(), global_step = step)
                self.writer.add_scalar("Gaze Estimator/mpiigaze valid mean diff gaze", self.mpiigaze_valid.item(), global_step = step)
                self.writer.add_scalar("Gaze Estimator/mpiigaze test mean diff gaze", self.mpiigaze_test.item(), global_step = step)          
                self.writer.add_scalar("Gaze Estimator/ETH-Xgaze valid mean diff gaze", self.eth_xgaze_valid.item(), global_step = step)
                # self.writer.add_scalar("Gaze Estimator/mpiigaze test calib mean diff gaze", self.mpiigaze_calib_test.item(), global_step = step)        
                # self.writer.add_scalar('Hourglass/synthetic gazemap loss', self.synthetic_gazemap_loss.item(), global_step=step)
                self.writer.add_scalar('Densenet/synthetic gaze loss', self.pred_refined_gaze_loss.item(), global_step=step)                                

                # for i in range(len(self.synthe_ldmks_diff)):
                #     key_name = None
                #     key_name = 'ldmk_' + str(i)
                #     self.writer.add_scalar("Landmarks/synthe_ldmk_diff" + str(i), self.synthe_ldmks_diff[key_name].item(), global_step = step)
                #     self.writer.add_scalar("Landmarks/synthe-refine_ldmk_diff" + str(i), self.synthe_refine_ldmks_diff[key_name].item(), global_step = step)

                self.writer.close()
                
            if step % config['Refiner1_save_interval']['num'] == 0:
                torch.save(self.R1.state_dict(), os.path.join(config['Refiner1_checkpoint_path']['pathname'], config['refiner1_path']['pathname'] % step))
            
            if step % config['save_interval']['num'] == 0:
                print('Saving checkpoints, Step : {}'.format(step))
                torch.save(self.R1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['refiner1_path']['pathname'] % step))
                torch.save(self.R2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['refiner2_path']['pathname'] % step))
                torch.save(self.G_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['global_discriminator1_path']['pathname'] % step))
                torch.save(self.G_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['global_discriminator2_path']['pathname'] % step))
                torch.save(self.L_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['local_discriminator1_path']['pathname'] % step))
                torch.save(self.L_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['local_discriminator2_path']['pathname'] % step))
                torch.save(self.Reg.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['regressor_path']['pathname'] % step))
                torch.save(self.Densenet.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['densenet_path']['pathname'] % step))


                state = {
                    'step': step,
                    'optG_D_1' : self.global_discriminator1_optimizer.state_dict(),
                    'optG_D_2' : self.global_discriminator2_optimizer.state_dict(),
                    'optL_D_1' : self.local_discriminator1_optimizer.state_dict(),
                    'optL_D_2' : self.local_discriminator2_optimizer.state_dict(),
                    'optR1' : self.refiner1_optimizer.state_dict(),
                    'optR2' : self.refiner2_optimizer.state_dict(),
                    'optReg' : self.regressor_optimizer.state_dict(),
                    'optDensenet' : self.Densenet_optimizer.state_dict()
                }
                
                torch.save(state, os.path.join(config['checkpoint_path']['pathname'], config['optimizer_path']['pathname'] % step))

if __name__ == '__main__':
	trainer = TrainGazeEstimator()
	trainer.train()