# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
# Comments added by Abhimanyu Talwar
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def BboxTransformInv(boxes, deltas, batch_size):
    """
    boxes and deltas are of shape (batch_size x num_anchors*feat_h*feat_w x 4).
    (1) boxes contains shifted anchor boxes covering the original image
    (2) deltas contains predictions of the RPN
    
    """
    
    # Each 4-tuple in boxes contains (x_min, y_min, x_max, y_max). That explains
    # the following calcuations.
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights
    
    # Note: Use of '::' is just to keep the dimension while slicing. Now coming
    # to what these deltas are:
    # (1) dx = (x - xa)/wa
    # (2) dy = (y - ya)/ha
    # (3) dw = log(w/wa)
    # (3) dh = log(h/ha)
    # Here variables x and xa refer to the predicted and anchor boxes 
    # respectively.
    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]
    
    # These calculations just follow from the definitions of deltas above.
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    # Note: clone() returns a copy of the self tensor but unlike copy(), the
    # gradients propogating to the cloned tensor will propogate to the original
    # tensor as well.
    pred_boxes = deltas.clone()
    # x_min
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y_min
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x_max
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y_max
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h
    
    return pred_boxes

def ClipBoxes(boxes, im_shape, batch_size):
    """
    boxes are the predicted box proposals (as returned by the BboxTransformInv
    function).
    """
    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes

def BboxOverlapsBatch(anchors, gt_boxes):
    """
    INPUTS:
    (1) anchors:  Shape (num_blocks*num_anchors x 4)
    (2) gt_boxes: Shape (batch_size, K, 5)
    """
    
    batch_size = gt_boxes.size(0)
    
    if anchors.dim() == 2:
    	N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1) 
        
