from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Comments added by Abhimanyu Talwar. 
# --------------------------------------------------------

import numpy as np
import pdb

def GenerateAnchors(base_size, anchor_ratios, anchor_scales):
    """
    The base_anchor is a square box of dimensions base_size.
    The value of base_size is chosen as the feature stride of
    the feature extractor network being used (35 for ShuffleNet).
    
    The aim of this function is to transform the base_anchor
    using all combination of anchor_scales and anchor_ratios.
    
    Each anchor is a rectanglt defined by a 4-tuple containing 
    coordinates (x_min, y_min, x_max, y_max) of two non-adjacent 
    vertices.
    
    """
    base_anchor = np.array([0, 0, base_size-1, base_size-1])
    ratioed_anchors = ApplyRatios(base_anchor, anchor_ratios)
    anchors = np.vstack([ApplyScales(anchor_, anchor_scales) \
                         for anchor_ in ratioed_anchors])
    return anchors
    
def ApplyRatios(anchor, anchor_ratios):
    """
    This assume that aspect_ratio is defined as height/width.
    Let the ratio be r, then assuming the width is w', the height
    is rw'. Then the area is rw'^2, and so given the area 
    and a ratio, we can calculate the corresponding w' and h'.
    
    P.S. It doesn't really matter whether ratio is defined as w/h or
    h/w because in the end we use a "symmetric" list of ratios, e.g.
    [0.5, 1.0, 2.0]
    
    """
    w, h, ctr_x, ctr_y = GetAnchorTuple(anchor)
    anchor_area = w*h
    w_arr = np.round(np.sqrt(anchor_area/anchor_ratios))
    h_arr = np.round(w_arr*anchor_ratios)
    
    # The shape of anchors will be len(anchor_ratios) x 4
    anchors = MakeAnchors(w_arr, h_arr, ctr_x, ctr_y)
    return anchors

def ApplyScales(anchor, anchor_scales):
    w, h, ctr_x, ctr_y = GetAnchorTuple(anchor)
    w_arr = w*anchor_scales
    h_arr = h*anchor_scales
    
    # The shape of anchors will be len(anchor_scales) x 4
    anchors = MakeAnchors(w_arr, h_arr, ctr_x, ctr_y)
    return anchors
    
def MakeAnchors(w_arr, h_arr, ctr_x, ctr_y):
    # We add a new axis because np.hstack concatenates along
    # the second dimension.
    w_arr = w_arr[:, np.newaxis]
    h_arr = h_arr[:, np.newaxis]
    
    # The shape of anchors will be len(w_arr) x 4
    anchors = np.hstack((ctr_x - 0.5*(w_arr - 1),
                        ctr_y - 0.5*(h_arr -1),
                        ctr_x + 0.5*(w_arr - 1),
                        ctr_y + 0.5*(h_arr -1)))
    return anchors
    
def GetAnchorTuple(anchor):
    """
    Returns height, width and center coordinates for an anchor.
    """
    h = anchor[3] - anchor[1] + 1
    w = anchor[2] - anchor[0] + 1
    ctr_x = anchor[0] + 0.5*(w-1)
    ctr_y = anchor[1] + 0.5*(h-1)
    return w, h, ctr_x, ctr_y

if __name__ == '__main__':
    # Test GenerateAnchors below
    _anchors = GenerateAnchors(base_size=16,
                              anchor_ratios=[0.5, 1.0, 2.0],
                              anchor_scales=2**np.arange(3,6))
    print(_anchors)
