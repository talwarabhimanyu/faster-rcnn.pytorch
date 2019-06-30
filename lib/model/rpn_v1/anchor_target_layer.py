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

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import GenerateAnchors
from .bbox_transform import ClipBoxes, BboxOverlapsBatch, BboxTransformBatch

import pdb

DEBUG = False

class AnchorTargetLayer(nn.Module):
    """
    Each anchor box is classified into one of three categories: positive, negative, and
    other. For loss computation, we do not use the 'other' category. The purpose of this
    layer is to assign labels to anchor boxes.

    The label assigned to an anchor box is ...
    
    """

    def __init__(self, feat_stride, anchor_scales, anchor_ratios):
        super(AnchorTargetLayer, self).__init__()
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchors = torch.from_numpy(GenerateAnchors(anchor_scales=np.array(anchor_scales), \
			anchor_ratios=np.array(ratios))).float()
        self.num_anchors = self.anchors.size(0)

        self.allowed_border = 0

    def forward(self, x):
        cls_score, gt_boxes, im_info, num_boxes = x
        
        """
        The next few lines do something similar to what is done in the ProposalLayer - see my
        detailed comments (with examples) in proposal_layer.py.

        In summary, we have num_anchors variants of the base anchor block (of dimensions 
        feat_stride x feat_stride) - with each variant being a combination of an anchor scale and
        anchor ratio. Now, if we translate these num_anchor base variations to each block
        of dimension feat_stride x feat_stride in the original image, then that whole set of
        anchors is what we finally store in _anchors below.

        """
        feat_h, feat_w = cls_score.shape[3], cls_score.shape[4]
        batch_size = gt_boxes.size(0)

	shift_x = np.arange(0, feat_w) * self.feat_stride
        shift_y = np.arange(0, feat_h) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(cls_score).float()
        
        num_anchors = self.anchors.shape[0]
        num_blocks = shift.shape[0]

        self.anchors = self.anchors.type_as(gt_boxes)
        _anchors = self.anchors.view(1, num_anchors, 4) + shifts.view(num_blocks, 1, 4)
        _anchors = _anchors.view(num_anchors * num_blocks, 4)

        total_anchors = int(num_blocks * num_anchors)

        # Get indices of anchors which lie inside the image.
        keep = ((anchors[:, 0] >= -self.allowed_border) &
                (anchors[:, 1] >= -self.allowed_border) &
                (anchors[:, 2] < int(im_info[0][1]) + self.allowed_border) &
                (anchors[:, 3] < int(im_info[0][0]) + self.allowed_border))
        
        inds_inside = torch.nonzero(keep).view(-1)

        # Keep only those anchors which are inside the image
        _anchors = _anchors[inds_inside, :]

        # Label: 1 is positive, 0 is negative, -1 is other
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)

        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        overlaps = BboxOverlapsBatch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        



