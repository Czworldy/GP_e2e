#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image

def get_cost_map(img, point_cloud, args):
    img2 = np.zeros((args.height, args.width), np.uint8)
    img2.fill(255)
    
    pixs_per_meter = args.height/args.scale
    u = (args.height-point_cloud[0]*pixs_per_meter).astype(int)
    v = (-point_cloud[1]*pixs_per_meter+args.width//2).astype(int)

    mask = np.where((u >= 0)&(u < args.height))[0]
    u = u[mask]
    v = v[mask]
    
    mask = np.where((v >= 0)&(v < args.width))[0]
    u = u[mask]
    v = v[mask]

    img2[u,v] = 0
    kernel = np.ones((17,17),np.uint8)  
    img2 = cv2.erode(img2,kernel,iterations = 1)
    
    kernel_size = (21, 21)
    img = cv2.dilate(img,kernel_size,iterations = 3)
    
    img = cv2.addWeighted(img,0.5,img2,0.5,0)
    
    mask = np.where((img2 < 50))
    u = mask[0]
    v = mask[1]
    img[u, v] = 0
    return img

def generate_costmap(ts, global_dict, cost_map_trans, args):
    cost_map = get_cost_map(global_dict['ipm_image'], global_dict['pcd'], args)
    global_dict['cost_map'] = cost_map
    # print('cost_map shape', cost_map.shape)
    img = Image.fromarray(cv2.cvtColor(cost_map,cv2.COLOR_BGR2RGB)).convert('L')
    trans_img = cost_map_trans(img)
    #global_dict['trans_costmap_dict'][ts] = trans_img # need to del
    global_dict['trans_costmap'] = trans_img
    # print('trans_img shape', trans_img.shape)
    global_dict['ts'] = ts
    
def find_nn_ts(ts_list, t):
    if len(ts_list) == 1: return ts_list[0]
    if t <= ts_list[0]: return ts_list[0]
    if t >= ts_list[-1]: return ts_list[-1]
    for i in range(len(ts_list)-1):
        if ts_list[i] < t and t < ts_list[i+1]:
            return ts_list[i] if t-ts_list[i] < ts_list[i+1]-t else ts_list[i+1]
    print('Error in find_nn_ts')
    return ts_list[-1]

def get_costmap_stack(global_dict):
    ts_list = [ ts for ts in list(global_dict['trans_costmap_dict'].keys())]
    ts_list.sort()
    t0 = max(ts_list)
    trans_costmap_stack = []
    use_ts_list = []
    for i in range(-9,1):
        ts = find_nn_ts(ts_list, t0 + 0.0375*3*i)
        trans_costmap = global_dict['trans_costmap_dict'][ts]
        trans_costmap_stack.append(trans_costmap)
        use_ts_list.append(t0-ts)
    #print(use_ts_list)
    for ts in ts_list:
        if t0 - ts > 5:
            del global_dict['trans_costmap_dict'][ts]
    return trans_costmap_stack