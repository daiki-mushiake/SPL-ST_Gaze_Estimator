import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from core.update import BasicUpdateBlock, SmallUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords0.requires_grad = True
        coords1.requires_grad = True
        
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def loss_normalizar(self, loss):
        # std_1ch = torch.std(loss[:, :1, 60:340, 60:550])
        # ave_1ch = torch.mean(loss[:, :1, 60:340, 60:550])
        min_1ch = torch.min(loss[:, :1, 60:340, 60:550])
        max_1ch = torch.max(loss[:, :1, 60:340, 60:550])

        # std_2ch = torch.std(loss[:, 1:2, 60:340, 60:550])
        # ave_2ch = torch.mean(loss[:, 1:2, 60:340, 60:550])
        min_2ch = torch.min(loss[:, 1:2, 60:340, 60:550])
        max_2ch = torch.max(loss[:, 1:2, 60:340, 60:550])
        
        # print("max_1ch",max_1ch)
        # print("max_2ch",max_2ch)
        # print("min_1ch",min_1ch)
        # print("min_2ch",min_2ch)

        """standardization"""
        # standardized_loss_1ch = (loss[:, :1, 60:340, 60:550] - ave_1ch)/std_1ch
        # standardized_loss_2ch = (loss[:, 1:2, 60:340, 60:550] - ave_2ch)/std_2ch

        """Normalization"""
        normalized_loss_1ch = (loss[:, :1, 60:340, 60:550] - min_1ch) / (max_1ch - min_1ch)
        normalized_loss_2ch = (loss[:, 1:2, 60:340, 60:550] - min_2ch) / (max_2ch - min_2ch)
        
        
        
        """Invert the pixel value"""
        normalized_img_1ch = 1 - normalized_loss_1ch
        normalized_img_2ch = 1 - normalized_loss_2ch

        normalized_img = (normalized_img_1ch + normalized_img_2ch)

        n_img_min = torch.min(normalized_img)
        n_img_max = torch.max(normalized_img)

        normalized_img = ((normalized_img - n_img_min) / (n_img_max - n_img_min)) * 255

        return normalized_img

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        deformation_loss = None

        image1 = image1.contiguous()
        image2 = image2.contiguous()    

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        #fmap1.register_hook(lambda grad: print("fmap1:",grad))
        #fmap2.register_hook(lambda grad: print("fmap2:",grad))

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            #inp.register_hook(lambda grad: print("inp:",grad))

        coords0, coords1 = self.initialize_flow(image1)
        #image1.register_hook(lambda grad: print("image1_raft:",grad))
        #coords0.register_hook(lambda grad: print("coords0:",grad))
        #coords1.register_hook(lambda grad: print("coords1:",grad))
        #print("coords1:",coords1)
        nan_checker_coords0 = torch.isnan(coords0)
        if True in nan_checker_coords0:
            assert True not in torch.isnan(coords0), 'Nan in coords0'
        nan_checker_coords1 = torch.isnan(coords1)
        if True in nan_checker_coords1:
            assert True not in torch.isnan(coords1), 'Nan in coords1'

        if flow_init is not None:
            coords1 = coords1 + flow_init
            

        flow_predictions = []
        for itr in range(iters):
            #coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            #corr.register_hook(lambda grad: print("corr:",grad))

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            #delta_flow.register_hook(lambda grad: print("delta_flow=",grad)) 

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            #print("Flow Up=",flow_up)
            #if itr == 1:
                #flow_up.register_hook(lambda grad: print("flow_up_" + str(itr)+"=",grad))
            
            flow_predictions.append(flow_up)
            #flow_predictions[itr].register_hook(lambda grad: print("flow_up_" + str(itr)+"=",grad))

        nan_checker = torch.isnan(flow_up)
        if True in nan_checker:
            assert True not in torch.isnan(flow_up), 'Nan in optical flow'         

        flow_img = self.loss_normalizar(flow_up)

        # deformation_loss_sum = flow_up[:, :1, 60:340, 60:550] + flow_up[:, 1:2, 60:340, 60:550]
        # deformation_loss_sum = torch.abs(deformation_loss_sum)
        # deformation_loss_sum_ave = torch.mean(deformation_loss_sum)
        # print(deformation_loss_tensor)
        # print(deformation_loss_inv)

        if test_mode:
            #return coords1 - coords0, flow_up
            # return deformation_loss_sum_ave * 0.1, flow_img
            return flow_up, flow_img
            
        return flow_predictions
