import torch
import torch.nn as nn
import numpy as np
import yaml

with open('config.yaml', 'r') as yml:
    config = yaml.load(yml,Loader=yaml.Loader)

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        #print("pred = ",pred)
        #print("gt = ",gt)
        loss = ((pred - gt)**2)
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


class GazeLoss:
    def __init__(self):
        self.cos = nn.CosineSimilarity()

    def __call__(self, gt_gaze, gaze):
        gt_gaze = gt_gaze.cuda()
        gt_gaze = np.squeeze(gt_gaze)
        gaze = gaze.cuda()
        one_tensor = torch.ones(gaze.shape[0]).cuda()

        normalized_gaze = torch.nn.functional.normalize(gaze, dim=2)#正規化
        normalized_gaze = torch.squeeze(normalized_gaze) 
        cos_sim = self.cos(gt_gaze, normalized_gaze)
        gaze_loss = (one_tensor - cos_sim)

        return gaze_loss
    
class LandmarkLoss:
    def __init__(self):
        self.heatmapLoss = HeatmapLoss()
        self.landmarks_loss = nn.MSELoss()      

    def __call__(self, heatmaps_pred, landmarks_pred, gt_heatmaps, gt_landmarks):
        combined_calc_loss = []
        self.gt_heatmaps = gt_heatmaps.cuda()
        self.gt_landmarks = gt_landmarks.cuda()

        for i in range(config['nstack']['param']):
            combined_calc_loss.append(self.heatmapLoss(heatmaps_pred[:, i, :], self.gt_heatmaps))
        heatmap_loss = torch.stack(combined_calc_loss, dim=1)
        landmark_loss = self.landmarks_loss(landmarks_pred, self.gt_landmarks)

        return heatmap_loss, landmark_loss