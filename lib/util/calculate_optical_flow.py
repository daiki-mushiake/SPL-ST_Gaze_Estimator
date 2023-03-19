import numpy as np
import torch.nn as nn
import torch
import cv2
import yaml
from core.utils.utils import InputPadder
from torchvision import utils
from core.utils import flow_viz
import core.mask_generator

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)


def up_scaler(imgs):
    up_scale_img = nn.Upsample(scale_factor=11, mode='bilinear')
    imgs = up_scale_img(imgs)

    return  imgs

def optical_flow_calc(image1, image2, RAFT, landmarks):
    
    """Initialize """
    deformation_loss = 0
    landmarks_mask = landmarks

    image1 = image1.repeat(1,3,1,1)
    image2 = image2.repeat(1,3,1,1)
    image1 = up_scaler(image1)
    image2 = up_scaler(image2)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    deformation, flow_img = RAFT(image1, image2, iters=10, test_mode=True)
    deformation_loss_sum = abs(deformation[:, :1, :, :]) + abs(deformation[:, 1:2, :, :])
    # masked_loss = deformation_loss_sum
    deformation_loss_ave = torch.mean(deformation_loss_sum)
    deformation_loss = deformation_loss_ave

    return deformation_loss, flow_img

def after_optical_flow_calc(image1, image2, RAFT, landmarks):
    
    """Initialize """
    deformation_loss = 0
    landmarks_mask = landmarks

    image1 = image1.repeat(1,3,1,1)
    image2 = image2.repeat(1,3,1,1)
    image1 = up_scaler(image1)
    image2 = up_scaler(image2)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    deformation, flow_img = RAFT(image1, image2, iters=10, test_mode=True)
    magnification_x = image1.shape[2] / 36 #MPIIGAZE & UT Multi-view
    magnification_y = image1.shape[3] / 60 #MPIIGAZE & UT Multi-view
    landmarks_mask[:,0:31,1] = landmarks_mask[:,0:31,1] * magnification_x
    landmarks_mask[:,0:31,0] = landmarks_mask[:,0:31,0] * magnification_y
    mask = core.mask_generator.mask_generator(deformation, landmarks_mask[:,0:15,:], landmarks_mask[:,16:31,:])
    # masked_img = image1 * mask
    # utils.save_image(masked_img, config['save_train_image_directory']['dirname'] + 'masked_img.png')

    deformation_loss_sum = deformation[:, :1, :, :] + deformation[:, 1:2, :, :]
    masked_loss = deformation_loss_sum * mask
    # masked_loss = deformation_loss_sum
    deformation_loss_masked = torch.masked_select(deformation_loss_sum, masked_loss > 0)
    deformation_loss_ave = torch.abs(deformation_loss_masked)
    deformation_loss_ave = torch.mean(deformation_loss_ave)
    deformation_loss = deformation_loss_ave * 0.1


    return deformation_loss, flow_img

def before_optical_flow_calc(image1, image2, RAFT, landmarks):
    
    """Initialize """
    deformation_loss = 0
    landmarks_mask = landmarks

    image1 = image1.repeat(1,3,1,1)
    image2 = image2.repeat(1,3,1,1)
    image1 = up_scaler(image1)
    image2 = up_scaler(image2)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    magnification_x = image1.shape[2] / 36 #MPIIGAZE & UT Multi-view
    magnification_y = image1.shape[3] / 60 #MPIIGAZE & UT Multi-view
    landmarks_mask[:,0:31,1] = landmarks_mask[:,0:31,1] * magnification_x
    landmarks_mask[:,0:31,0] = landmarks_mask[:,0:31,0] * magnification_y
    mask = core.mask_generator.mask_generator(image1, landmarks_mask[:,0:15,:], landmarks_mask[:,16:31,:])
    image1 = image1 * mask
    image2 = image2 * mask
    deformation, flow_img = RAFT(image1, image2, iters=10, test_mode=True)

    deformation_loss_sum = deformation[:, :1, :, :] + deformation[:, 1:2, :, :]
    deformation_loss_masked = torch.masked_select(deformation_loss_sum, mask > 0)
    deformation_loss_ave = torch.abs(deformation_loss_masked)
    deformation_loss_ave = torch.mean(deformation_loss_ave)
    deformation_loss = deformation_loss_ave * 0.1

    return deformation_loss, flow_img

def flow_visualizer(flow_img):
    flow_up_img_list = []
    flow_up_imgs = flow_img.permute(0,2,3,1).detach().cpu().numpy()
    for i in range(flow_up_imgs.shape[0]):
        flow_up_img_list.append(flow_viz.flow_to_image(flow_up_imgs[i]))
    flow_up_img = np.array(flow_up_img_list)
    flow_up_img = flow_up_img.transpose(0,3,1,2) 
    flow_up_img = torch.from_numpy(flow_up_img.astype(np.float32)).clone().cuda()
    flow_up_img = flow_up_img/255

    return flow_up_img

def img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images, real_images, edge_synthetic_images, edge_cg_to_real_images, flow_img, img_name, cnt):
    output_list = [synthetic_images, cg_to_real_images, return_real_to_cg_images, real_images, edge_synthetic_images, edge_cg_to_real_images]
    cat_images = torch.cat(output_list, dim = 0)
    cat_images = utils.make_grid(cat_images.cuda(),3)
    utils.save_image(cat_images, config['save_train_image_directory']['dirname'] + img_name + str(cnt) + '.png')

    normalize_array = flow_img.to('cpu').detach().numpy().copy()
    normalize_img = normalize_array[0].transpose(1, 2, 0)

    # cv2.imwrite("../mpiigaze_train_image/normalize_img_"+ img_name + "_" + str(cnt)+".jpg",normalize_img)

def save_refine_img(current_step, pretrain_step, synthetic_data_iter, real_data_iter, edge_detection, R1, R2, RAFT, pretrain=False):
    synthetic_images, _, landmarks, _, _ = next(synthetic_data_iter)
    edge_synthetic_images = edge_detection(synthetic_images)
    real_images = next(real_data_iter)
    cg_to_real_images = R1(synthetic_images)
    edge_cg_to_real_images = edge_detection(cg_to_real_images)
    return_real_to_cg_images = R2(cg_to_real_images)
    
    synthetic_images = synthetic_images.cuda()
    cg_to_real_images = cg_to_real_images.cuda()
    real_images = real_images.cuda()
    edge_synthetic_images = edge_synthetic_images.cuda()
    edge_cg_to_real_images = edge_cg_to_real_images.cuda()
    return_real_to_cg_images = return_real_to_cg_images.cuda()

    # _, flow_img = optical_flow_calc(synthetic_images, cg_to_real_images, RAFT, landmarks)
    # import pdb;pdb.set_trace()
    mask = core.mask_generator.mask_generator(synthetic_images, landmarks[:,0:14,:], landmarks[:,16:31,:])
    masked_img = (synthetic_images*255) * mask
    # import pdb;pdb.set_trace()
    # flow_img = mask*255
    # img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images, real_images, edge_synthetic_images, edge_cg_to_real_images, flow_img, 'train_step', current_step)
    img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images, real_images, edge_synthetic_images, edge_cg_to_real_images, masked_img, 'train_step', current_step)

    # if not pretrain:
    #     if current_step % 100 == 0:
    #         # img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images, real_images, edge_synthetic_images, edge_cg_to_real_images, flow_img, 'train_step', current_step)
    #         img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images, real_images, edge_synthetic_images, edge_cg_to_real_images, masked_img, 'train_step', current_step)
    # else:
    #     if pretrain_step < config['refiner_pretrain_iteration']['num']:
    #         # img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images,real_images, edge_synthetic_images, edge_cg_to_real_images, flow_img, 'pretrain_step', pretrain_step)
    #         img_visualizer(synthetic_images, cg_to_real_images, return_real_to_cg_images,real_images, edge_synthetic_images, edge_cg_to_real_images, masked_img, 'pretrain_step', pretrain_step)
    #         print("self.pretrain_step:",pretrain_step)
    #         pretrain_step += 1
