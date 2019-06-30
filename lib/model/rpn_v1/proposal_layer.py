from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
# --------------------------------------------------------
# Comments added by Abhimanyu Talwar.
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    def __init__(self, feat_stride, anchor_scales, anchor_ratios):
        super(_ProposalLayer, self).__init__()
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        # Shape of anchors is len(anchor_scales)*len(anchor_ratios) x 4 where
        # 4 coordinates parametrize an anchor: (x_min, y_min, x_max, ymax).
        self.anchors = torch.from_numpy(GenerateAnchors(
                                        scales=np.array(anchor_scales),
                                        ratios=np.array(anchor_ratios))).float()
        
    def forward(self, x):
        """
        The input 'x' here is a tuple (cls_score, bbox_deltas, im_shape, config):
        (1) cls_score has shape (batch_size x num_objects x num_anchors x 
            feat_h x feat_w)
        (2) bbox_deltas has shape (batch_size x 4*num_anchors x feat_h x feat_w)
        """
        cls_score, bbox_deltas, im_shape, config = x
        
        foreground_score = cls_score[:,1,:,:,:]
        batch_size = bbox_deltas.shape[0]
        
        post_nms_topN = config['RPN_POST_NMS_TOP_N']
        pre_nms_topN = config['RPN_PRE_NMS_TOP_N']
        nms_thresh = config['RPN_NMS_THRESH']
        min_size = config['RPN_MIN_SIZE']
        
        
        """
        An example of what the next few lines of code are doing:
        Say feat_w, feat_h is (2, 3) and stride is 2. Then we have:
        
        shift_x.ravel() = [0, 2, 0, 2, 0, 2]
        shift_y.ravel() = [0, 0, 2, 2, 4, 4]
        
        Note that the original image may have been of size (6,4) that is, 
        (feat_h*stride, feat_w*stride). Therefore if we iterate over the 
        coordinates given by zip(shift_x.ravel(), shift_y.ravel()), then
        we will be iterating over blocks of the original image, starting with
        (0,0) and moving with a stride 2. The number of such blocks (num_blocks)
        is feat_h*feat_w
        
        
        Looking at shifts:
        
        shifts = [[0, 0, 0, 0],
                  [2, 0, 2, 0],
                  [0, 2, 0, 2],
                  [2, 2, 2, 2],
                  [0, 4, 0, 4],
                  [2, 4, 2, 4]]
        
        The shape of 'shifts' is (num_blocks x 4). In each row, the first two
        values denote the first cell of a block in the original image. The 3rd
        and 4th values are just the first two values repeated.
        
        """
        feat_h, feat_w = cls_score.shape[3], cls_score.shape[4]
        shift_x = np.arange(0, feat_w) * self.feat_stride
        shift_y = np.arange(0, feat_h) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(foreground_score).float()
        
        num_anchors = self.anchors.shape[0]
        num_blocks = shift.shape[0]
        
        # This step will result in _anchors being of shape (num_blocks x
        # num_anchors x 4). The result of this broadcasted addition is that
        # the 4-tuple associated with each of the anchors, will be added 
        # elementwise to the 4-tuple associated with each of the blocks.
        #
        # Since each row of 'shifts' is the two indices (repeated twice) of 
        # the first cell of each of the num_blocks blocks in the original image,
        # by adding a row of 'shifts' to a row of 'self.anchors' (which contains
        # coordinates of the two non-adjacent points of the anchor rectangle),
        # we are merely translating the anchor rectangle. 
        _anchors = self.anchors.view(1, num_anchors, 4) + \
                        shifts.view(num_blocks, 1, 4)
        
        # We first reshape the (num_blocks x num_anchors x 4) shaped _anchors
        # to the shape (1, num_blocks * num_anchors, 4). After the expand 
        # operation, the shape is (batch_size, num_blocks * num_anchors, 4).
        _anchors = _anchors.view(1, num_blocks * num_anchors, 4).\
                        expand(batch_size, num_blocks * num_anchors, 4)
        
        # After permutation and view, shape of bbox_deltas becomes (batch_size x 
        # feat_h*feat_w*num_anchors x 4). Same for foreground_score. Since 
        # feat_h*feat_w equals num_blocks, the shape is the same as _anchors'.
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        
        foreground_score = foreground_score.permute(0, 2, 3, 1).contiguous()
        foreground_score = foreground_score.view(batch_size, -1)
        
        # Shape of all inputs is (batch_size x num_anchors*feat_h*feat_w x 4).
        # We use predictions (bbox_deltas) and anchor box templates to compute
        # predicted box proposals.
        proposals = BboxTransformInv(_anchors, bbox_deltas, batch_size)
        
        # Clip boxes to fit within image boundaries
        proposals = ClipBoxes(proposals, im_info, batch_size)
        
        # Sort scores in descending order
        scores_keep = foreground_score
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)
        
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        
        # We process each image in the batch individually. Specifically, for an
        # image, we:
        # (1) first select the top pre_nms_topN proposals, as ranked by their
        #     foreground score.
        # (2) apply Non Maximum Suppression (NMS) to get rid of duplicate boxes,
        # (3) select the top post_nms_topN proposals from the proposals 
        #     remaining after NMS. 
        for i in range(batch_size):
            # (1) Select the top pre_nms_topN proposals for this image.
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i] 
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)
            
            # (2) Apply NMS
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), \
                               1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)
            
            # (3) Keep the top post_nms_topN proposals
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]
            
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single
        return output
