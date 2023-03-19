import torch
import torch.nn as nn
import yaml

with open('config.yaml', 'r') as yml:
    config = yaml.safe_load(yml)

def calc_iris_parameter(ldmk):
    
    idx_sub = torch.arange(ldmk.shape[0])

    '''Find x,y_max&min'''
    # iris_center = ldmk[idx_sub,32]
    iris_center = torch.mean(ldmk[idx_sub, 16:32],1)
    iris_ldmk = ldmk[idx_sub, 16:32]
    x_iris = ldmk[:,16:32,1]
    x_max_idx = torch.max(x_iris, 1)
    x_min_idx = torch.min(x_iris, 1)    

    y_iris = ldmk[:,16:31,0]
    y_max_idx = torch.max(y_iris, 1)
    y_min_idx = torch.min(y_iris, 1)

    x_max = x_iris[idx_sub, x_max_idx[1]]
    x_min = x_iris[idx_sub, x_min_idx[1]]
    y_max = y_iris[idx_sub, y_max_idx[1]]
    y_min = y_iris[idx_sub, y_min_idx[1]]

    right_corner = x_max
    left_corner = x_min
    top = y_min
    bottom = y_max

    radius_x = right_corner- iris_center[:,1]
    radius_y = bottom - iris_center[:,0]

    """Calculate Tilt"""
    vec_std = torch.tensor([1,0])
    vec_std = vec_std.expand(iris_center.shape[0],2).cuda()
    vec_eye = iris_ldmk[idx_sub, x_max_idx[1]] - iris_ldmk[idx_sub, x_min_idx[1]]
    vec_eye = vec_eye.cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    tilt = cos(vec_std, vec_eye)

    return radius_x, radius_y, tilt, iris_center

def calc_eyeball_parameter(ldmk, imgs):
    eyeball_center = ldmk[:,33]
    eyeball_radius = torch.tensor((1.1 * imgs.shape[2]) / 2).repeat(ldmk.shape[0])
    eyeball_tilt = torch.zeros(ldmk.shape[0]).cuda()

    return eyeball_radius, eyeball_radius, eyeball_tilt, eyeball_center

def create_map(img, center, rad_x, rad_y, tilt):
    x_map = torch.arange(img.shape[3]).reshape(1,-1)
    x_map = x_map.expand(img.shape[2], img.shape[3])
    x_map = torch.unsqueeze(x_map, 0)
    x_map = x_map.repeat(img.shape[0], 1, 1, 1).cuda()
    
    y_map = torch.arange(img.shape[2]).reshape(-1,1)
    y_map = img.shape[2] - y_map
    y_map = y_map.expand(img.shape[2], img.shape[3])
    y_map = torch.unsqueeze(y_map, 0)
    y_map = y_map.repeat(img.shape[0], 1, 1, 1).cuda()
    
    x0 = center[:, 1]
    x0 = torch.reshape(x0, (img.shape[0], 1))
    x0 = torch.unsqueeze(x0, 2)
    x0 = torch.unsqueeze(x0, 2).cuda()
    
    y0 = img.shape[2] - center[:,0]
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
    mask_img = torch.where(mask < 0, 1.0, 0.0)

    return mask_img

def gazemap_generator(imgs, ldmks):
    iris_radius_x, iris_radius_y, iris_tilt, iris_center = calc_iris_parameter(ldmks)
    iris_map_img = create_map(imgs, iris_center, iris_radius_x, iris_radius_y, iris_tilt)
    eyeball_radius_x, eyeball_radius_y, eyeball_tilt, eyeball_center = calc_eyeball_parameter(ldmks, imgs)
    eyeball_map_img = create_map(imgs, eyeball_center, eyeball_radius_x, eyeball_radius_y, eyeball_tilt)
    gazemap = torch.cat((iris_map_img, eyeball_map_img), dim=1)
    
    return gazemap
