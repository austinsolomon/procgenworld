#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import math
import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, ws_map=None):
    mse_map = ((img1 - img2)) ** 2
    if ws_map is None:
        mse = mse_map.view(img1.shape[0], -1).mean(1, keepdim=True)
        
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    else:
        mse = mse_map.view(img1.shape[0], -1).mean(1, keepdim=True)
        ws_mse = (mse_map * ws_map).view(img1.shape[0], -1).mean(1, keepdim=True) / ws_map.mean()

        return 20 * torch.log10(1.0 / torch.sqrt(mse)), 20 * torch.log10(1.0 / torch.sqrt(ws_mse))


def genERP(i,j,N):
    val = math.pi/N
    # w_map[i+j*w] = cos ((j - (h/2) + 0.5) * PI/h)
    w = math.cos( (j - (N/2) + 0.5) * val )
    return w

def compute_map_ws(img):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = np.zeros((img.shape[0],img.shape[1]))

    for j in range(0,equ.shape[0]):
        for i in range(0,equ.shape[1]):
            equ[j,i] = genERP(i,j,equ.shape[0])

    return equ

def getGlobalWSMSEValue(mx,my):

    mw = compute_map_ws(mx)
    val = np.sum( np.multiply((mx-my)**2,mw) )
    den = val / np.sum(mw)

    return den

def ws_psnr(image1,image2):
    
    ws_mse   = getGlobalWSMSEValue(image1,image2)
    # second estimate the ws_psnr 

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf
    print("WS-PSNR ",ws_psnr)

    return ws_psnr
