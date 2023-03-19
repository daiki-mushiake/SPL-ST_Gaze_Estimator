
import torch
import torch.nn.functional as F
from torchvision import transforms,utils, models
import cv2
import numpy as np
import yaml
import torch.nn as nn
with open('config.yaml', 'r') as yml:
    config = yaml.load(yml,Loader=yaml.Loader)

def nan_checker(data, name):
    nan_check = torch.isnan(data)
    if True in nan_check:
        print("Nan in" +  name)
        print(name + ":" + data)
        data_name = name
        assert True not in torch.isnan(data), 'Nan in {[0]}'.format(data_name)

# def csv_create(current_step, synth_grad_imgs, cg_to_real_grad_imgs):
#     if current_step % 100 == 0 & current_step != 0:
#         synth_txtfname = '../grad_log'
#         synth_grad_imgs = synth_grad_imgs.to('cpu').detach().numpy().copy()
#         synth_grad_imgs = np.reshape(synth_grad_imgs, [-1, 60])
#         np.savetxt(synth_txtfname, synth_grad_imgs, fmt='%.5e')

#         cg_to_real_txtfname = config['cg_to_real_gradient_graph_txt']['txt_name']
#         cg_to_real_grad_imgs = cg_to_real_grad_imgs.to('cpu').detach().numpy().copy()
#         cg_to_real_grad_imgs = np.reshape(cg_to_real_grad_imgs, [-1, 60])
#         np.savetxt(cg_to_real_txtfname, cg_to_real_grad_imgs, fmt='%.5e')

def filter2d(imgs, kernel_w, kernel_h):
    grad_imgs = None

    imgs = imgs.cuda()
    imgs = imgs * 256

    kernel_w = torch.reshape(kernel_w,(1,1,3,3)).cuda()
    kernel_h = torch.reshape(kernel_h,(1,1,3,3)).cuda()

    grad_imgs_w = F.conv2d(imgs, kernel_w, padding=1)
    grad_imgs_h = F.conv2d(imgs, kernel_h, padding=1)

    return grad_imgs_w, grad_imgs_h

def normalizer(imgs_grad_value):
    imgs_grad_value = imgs_grad_value.cuda()
    numerator = None
    denominator = None 
    normalized_value = None

    numerator = imgs_grad_value - torch.min(imgs_grad_value, dim=3, keepdim=True)[0]
    denominator = torch.max(imgs_grad_value, dim = 3, keepdim=True)[0] - torch.min(imgs_grad_value, dim = 3, keepdim=True)[0] + 0.001
    normalized_value = numerator/ denominator

    nan_checker(normalized_value, 'normalized_value')

    return normalized_value

def grad_feature_loss(current_step, synth_imgs, cg_to_real_imgs):

    '''Initialize'''
    norm_synthe_w  = None
    norm_cg_to_real_w = None
    norm_synthe_h = None
    norm_cg_to_real_h = None
    loss = nn.MSELoss()

    '''Make Filter'''
    filter_w = torch.tensor([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]]).float()

    filter_h = torch.tensor([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]]).float()
    
    '''Sorvel Filter'''
    synthe_grad_w, synthe_grad_h = filter2d(synth_imgs, filter_w, filter_h)
    cg_to_real_grad_w, cg_to_real_grad_h = filter2d(cg_to_real_imgs, filter_w, filter_h)

    nan_checker(synthe_grad_w, 'synthe_grad_w')
    nan_checker(synthe_grad_h,'synthe_grad_h')
    nan_checker(cg_to_real_grad_w,'cg_to_real_grad_w')
    nan_checker(cg_to_real_grad_h, 'cg_to_real_grad_h')

    '''Gradient Normalize'''
    norm_synthe_w = normalizer(synthe_grad_w)
    norm_cg_to_real_w = normalizer(cg_to_real_grad_w)
    norm_synthe_h = normalizer(synthe_grad_h)
    norm_cg_to_real_h = normalizer(cg_to_real_grad_h)

    # '''Output Gradient Image'''
    # if current_step % 1000 == 0:
    #     synthe_w = '../grad_log/synthe_w' + str(current_step) + '.txt'
    #     cg_to_real_w = '../grad_log/cg_to_real_w' + str(current_step) + '.txt'
    #     synthe_h = '../grad_log/synthe_h' + str(current_step) + '.txt'
    #     cg_to_real_h = '../grad_log/cg_to_real_h' + str(current_step) + '.txt'

    #     norm_synthe_w_array = norm_synthe_w.to('cpu').detach().numpy().copy()
    #     norm_cg_to_real_w_array = norm_cg_to_real_w.to('cpu').detach().numpy().copy()
    #     norm_synthe_h_array = norm_synthe_h.to('cpu').detach().numpy().copy()
    #     norm_cg_to_real_h_array = norm_cg_to_real_h.to('cpu').detach().numpy().copy()

    #     norm_synthe_w_array = norm_synthe_w_array.reshape(config['batch_size']['size']*36,60)
    #     norm_cg_to_real_w_array = norm_cg_to_real_w_array.reshape(config['batch_size']['size']*36,60)
    #     norm_synthe_h_array = norm_synthe_h_array.reshape(config['batch_size']['size']*36,60)
    #     norm_cg_to_real_h_array = norm_cg_to_real_h_array.reshape(config['batch_size']['size']*36,60)

    #     synth_imgs_resize = torch.reshape(synth_imgs, (config['batch_size']['size']*36, 60))
    #     cg_to_real_imgs_resize = torch.reshape(cg_to_real_imgs, (config['batch_size']['size']*36, 60))

    #     utils.save_image(synth_imgs_resize, '../grad_log/synthe_' + str(current_step) + '.png')
    #     utils.save_image(cg_to_real_imgs_resize, '../grad_log/cg_to_real_' + str(current_step) + '.png')

    #     np.savetxt(synthe_w, norm_synthe_w_array, fmt='%.5e')
    #     np.savetxt(cg_to_real_w, norm_cg_to_real_w_array, fmt='%.5e')
    #     np.savetxt(synthe_h, norm_synthe_h_array, fmt='%.5e')
    #     np.savetxt(cg_to_real_h, norm_cg_to_real_h_array, fmt='%.5e')

    nan_checker(norm_synthe_w,'norm_synthe_w')
    nan_checker(norm_cg_to_real_w, 'norm_cg_to_real_w')
    nan_checker(norm_synthe_h, 'norm_synthe_h')
    nan_checker(norm_cg_to_real_h, 'norm_cg_to_real_h')

    # csv_create(current_step, norm_synthe_w, cg_to_real_grad_w)

    '''Calculate Graph Loss (MSE)'''
    graph_loss_w = loss(norm_synthe_w, norm_cg_to_real_w)
    graph_loss_h = loss(norm_synthe_h, norm_cg_to_real_h)
    
    graph_loss = torch.abs(graph_loss_w) + torch.abs(graph_loss_h)

    return torch.mean(graph_loss)
