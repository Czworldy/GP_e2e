#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CostMapDataset(Dataset):
    def __init__(self, data_index, opt, dataset_path='/media/wang/DATASET/CARLA/town01/', evalmode=False):
        self.points_num = opt.points_num
        self.evalmode = evalmode
        self.data_index = data_index
        self.weights = []
        self.max_dist = opt.max_dist
        self.max_speed = opt.max_speed
        self.max_t = opt.max_t
        self.img_step = opt.img_step
        transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            ]
        
        self.transform = transforms.Compose(transforms_)
        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        #self.acc_dict = {}
        self.files_dict = {}
        self.balance_dict = {}
        self.total_len = 0
        self.eval_index = None # eval mode
        self.eval_cnt = 0 # eval mode
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
            #self.read_acc(index)
            self.read_img(index)
            self.weights.append(len(self.files_dict[index]))
            self.data_balance()
        
    def data_balance(self):
        for data_index in list(self.files_dict.keys()):
            balance_dict = []
            for index in range(300, len(self.files_dict[data_index])-120):
            #for ts in self.files_dict[data_index][300:-120]:
                ts = self.files_dict[data_index][index]
                after_ts = self.files_dict[data_index][index+20]
                x_0 = self.pose_dict[data_index][ts][0]
                y_0 = self.pose_dict[data_index][ts][1]
                yaw = np.deg2rad(self.pose_dict[data_index][ts][3])
                x, y = self.tf_pose(data_index, after_ts, yaw, x_0, y_0)
                
                #if abs(self.pose_dict[data_index][after_ts][1]) > 1.0 and abs(self.vel_dict[data_index][after_ts][1]) > 1.0:
                if abs(y) > 0.5:
                    balance_dict.append(ts)
            self.balance_dict[data_index] = balance_dict
        
    def read_pose(self, index):
        file_path = self.dataset_path+str(index)+'/state/pos.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                x = float(sp_line[1])
                y = float(sp_line[2])
                z = float(sp_line[3])
                yaw = float(sp_line[5])
                ts_dict[ts] = [x, y, z, yaw]
        self.pose_dict[index] = ts_dict
        
    def read_vel(self, index):
        file_path = self.dataset_path+str(index)+'/state/vel.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                vx = float(sp_line[1])
                vy = float(sp_line[2])
                vz = float(sp_line[3])
                ts_dict[ts] = [vx, vy, vz]
        self.vel_dict[index] = ts_dict
        
    def read_acc(self, index):
        file_path = self.dataset_path+str(index)+'/state/acc.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                ax = float(sp_line[1])
                ay = float(sp_line[2])
                az = float(sp_line[3])
                ts_dict[ts] = [ax, ay, az]
        self.acc_dict[index] = ts_dict
    
    def read_img(self, index):
        files = glob.glob(self.dataset_path+str(index)+'/ipm/*.png')
        file_names = []
        for file in files:
            file_name = file.split('/')[-1][:-4]
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names

    def tf_pose(self, data_index, ts, yaw, x_0, y_0):
        x_t = self.pose_dict[data_index][ts][0]
        y_t = self.pose_dict[data_index][ts][1]
        dx = x_t - x_0
        dy = y_t - y_0
        x = np.cos(yaw)*dx + np.sin(yaw)*dy
        y = np.cos(yaw)*dy - np.sin(yaw)*dx
        return x, y
    
    def __len__(self):
        return 100000000000
    
    def __getitem__(self, index):
        while True:
            if self.evalmode:
                if self.eval_index == None:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
                data_index = self.eval_index
                file_name = self.files_dict[data_index][self.cnt]
                self.cnt += 20
                if self.cnt > len(self.files_dict[data_index])-20:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
            else:
                data_index = random.choices(self.data_index, self.weights)[0]
                file_name = random.sample(self.files_dict[data_index][300:-120], 1)[0]
                if random.random() < 0.5:
                    file_name = random.sample(self.balance_dict[data_index], 1)[0]
                
            ts_index = self.files_dict[data_index].index(file_name)
            
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            ts_list = []
            relative_t_list = []
            x_list = []
            y_list = []
            vx_list = []
            vy_list = []
            a_list = []
            for i in range(ts_index, len(self.files_dict[data_index])-100):
                ts = self.files_dict[data_index][i]
                if float(ts)-float(file_name) > self.max_t:
                    break
                else:
                    x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                    x_list.append(x_)
                    y_list.append(y_)
                    vx_ = self.vel_dict[data_index][ts][0]
                    vy_ = self.vel_dict[data_index][ts][1]
                    vx = np.cos(yaw)*vx_ + np.sin(yaw)*vy_
                    vy = np.cos(yaw)*vy_ - np.sin(yaw)*vx_
                    """
                    ax_ = self.acc_dict[data_index][ts][0]
                    ay_ = self.acc_dict[data_index][ts][1]
                    ax = ax_*np.cos(yaw) + ay_*np.sin(yaw)
                    ay = ay_*np.cos(yaw) - ax_*np.sin(yaw)
                    theta_a = np.arctan2(ay, ax)
                    theta_v = np.arctan2(vy, vx)
                    sign = np.sign(np.cos(theta_a-theta_v))
                    a = sign*np.sqrt(ax*ax + ay*ay)
                    a_list.append(a)
                    """
                    vx_list.append(vx)
                    vy_list.append(vy)
                    ts_list.append(ts)
                    relative_t_list.append(float(ts)-float(file_name))
                        
            if len(ts_list) == 0:
                continue
            else:
                #ts = random.sample(ts_list, 1)[0]
                #ts_array = random.sample(ts_list, self.points_num)
                #print(len(ts_list))
                if len(ts_list) < 6*(self.points_num-1)+1:
                    continue
                ts_array = [ts_list[6*item] for item in range(self.points_num)]
                
                #ts_array.sort()
                #weights = [np.exp(-0.23*(float(ts)-float(file_name))) for ts in ts_list]
                #sample_ts = random.choices(ts_list, weights)[0]
                #print(weights/sum(weights))
                break
            
        #ts = sample_ts
        # [0 ~ 1]
        # v0
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        v_0 = np.sqrt(_vx_0*_vx_0 + _vy_0*_vy_0)
        v_0 = torch.FloatTensor([v_0])
        v0_array = [v_0]*self.points_num

        t_array = []
        xy_array = []
        vxy_array = []
        """
        axy_array = []
        a_array = []
        """
        mirror = random.random() < 0.5
        for ts in ts_array:
            t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
            t_array.append(t)
            # x, y
            x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
            if not mirror:
                xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])# [-1, 1]
            else:
                xy = torch.FloatTensor([x/self.max_dist, -y/self.max_dist])
            xy_array.append(xy)
            #print('xy', xy, t)
            # yaw_t
            #yaw_t = angle_normal(np.deg2rad(self.pose_dict[data_index][ts][3]) - yaw)
            #yaw_t = torch.FloatTensor([yaw_t/np.pi])# [-1, 1]
            
            # vx, vy
            _vx = self.vel_dict[data_index][ts][0]
            _vy = self.vel_dict[data_index][ts][1]
            vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
            vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
            if not mirror:
                vxy_array.append(torch.FloatTensor([vx, vy]))
            else:
                vxy_array.append(torch.FloatTensor([vx, -vy]))
            
            # ax, ay
            """
            _ax = self.acc_dict[data_index][ts][0]
            _ay = self.acc_dict[data_index][ts][1]
            ax = _ax*np.cos(yaw) + _ay*np.sin(yaw)
            ay = _ay*np.cos(yaw) - _ax*np.sin(yaw)
            axy_array.append(torch.FloatTensor([ax, ay]))
            
            theta_a = np.arctan2(_ay, _ax)

            theta_v = np.arctan2(_vy, _vx)
            sign = np.sign(np.cos(theta_a-theta_v))
            a = sign*np.sqrt(ax*ax + ay*ay)
            a_array.append(a)
            """
        t = torch.FloatTensor(t_array)
        v_0 = torch.FloatTensor(v_0/self.max_speed)
        v0_array = torch.FloatTensor(v0_array)/self.max_speed
        xy = torch.stack(xy_array)
        #print(xy)
        vxy = torch.stack(vxy_array)
        """
        axy = torch.stack(axy_array)
        a = torch.FloatTensor(a_array)
        """
        
        #vxy = torch.FloatTensor([vx, vy])
        #axy = torch.FloatTensor([ax, ay])
        x_list = torch.FloatTensor(x_list)
        y_list = torch.FloatTensor(y_list)
        vx_list = torch.FloatTensor(vx_list)
        vy_list = torch.FloatTensor(vy_list)
        """
        a_list = torch.FloatTensor(a_list)
        """
        relative_t_list = torch.FloatTensor(relative_t_list)
        
        if self.evalmode:
            return {'t': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0,
                    #'a_list':a_list, 'axy':axy, 'a':a,
                    'x_list':x_list, 'y_list':y_list,
                    'vx_list':vx_list, 'vy_list':vy_list,
                    'ts_list':relative_t_list}
        else:
            #return {'t': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'v_0':v_0, 'v0_array':v0_array}
            return {'t': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0, 'v0_array':v0_array}
        

"""
class CostMapDataset_ORG(Dataset):
    def __init__(self, data_index, opt, dataset_path='/media/wang/DATASET/CARLA/town01/', evalmode=False):
        self.traj_steps = 8
        self.evalmode = evalmode
        self.data_index = data_index
        self.weights = []
        self.max_dist = opt.max_dist
        self.max_t = opt.max_t
        self.img_step = opt.img_step
        transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            ]
        
        self.transform = transforms.Compose(transforms_)
        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        self.acc_dict = {}
        self.files_dict = {}
        self.total_len = 0
        self.eval_index = None # eval mode
        self.eval_cnt = 0 # eval mode
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
            self.read_acc(index)
            self.read_img(index)
            self.weights.append(len(self.files_dict[index]))
        
    def read_pose(self, index):
        file_path = self.dataset_path+str(index)+'/state/pos.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                x = float(sp_line[1])
                y = float(sp_line[2])
                z = float(sp_line[3])
                yaw = float(sp_line[5])
                ts_dict[ts] = [x, y, z, yaw]
        self.pose_dict[index] = ts_dict
        
    def read_vel(self, index):
        file_path = self.dataset_path+str(index)+'/state/vel.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                vx = float(sp_line[1])
                vy = float(sp_line[2])
                vz = float(sp_line[3])
                ts_dict[ts] = [vx, vy, vz]
        self.vel_dict[index] = ts_dict
        
    def read_acc(self, index):
        file_path = self.dataset_path+str(index)+'/state/acc.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                ax = float(sp_line[1])
                ay = float(sp_line[2])
                az = float(sp_line[3])
                ts_dict[ts] = [ax, ay, az]
        self.acc_dict[index] = ts_dict
    
    def read_img(self, index):
        files = glob.glob(self.dataset_path+str(index)+'/ipm/*.png')
        file_names = []
        for file in files:
            file_name = file.split('/')[-1][:-4]
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names

    def tf_pose(self, data_index, ts, yaw, x_0, y_0):
        x_t = self.pose_dict[data_index][ts][0]
        y_t = self.pose_dict[data_index][ts][1]
        dx = x_t - x_0
        dy = y_t - y_0
        x = np.cos(yaw)*dx + np.sin(yaw)*dy
        y = np.cos(yaw)*dy - np.sin(yaw)*dx
        return x, y
    
    def __len__(self):
        return 100000000000
    
    def __getitem__(self, index):
        while True:
            if self.evalmode:
                if self.eval_index == None:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
                data_index = self.eval_index
                file_name = self.files_dict[data_index][self.cnt]
                self.cnt += 20
                if self.cnt > len(self.files_dict[data_index])-50:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
            else:
                data_index = random.choices(self.data_index, self.weights)[0]
                file_name = random.sample(self.files_dict[data_index][300:-120], 1)[0]
            ts_index = self.files_dict[data_index].index(file_name)
            imgs = []
            try:
                for i in range(-9,1):
                    _file_name = self.files_dict[data_index][ts_index + self.img_step*i]
                    image_path = self.dataset_path + str(data_index)+'/ipm/'+_file_name+'.png'
                    img = Image.open(image_path).convert('L')
                    img = self.transform(img)
                    imgs.append(img)
            except:
                print('get img error:', image_path)
                continue
            imgs = torch.stack(imgs)
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            ts_list = []
            relative_t_list = []
            x_list = []
            y_list = []
            vx_list = []
            vy_list = []
            a_list = []
            for i in range(ts_index, len(self.files_dict[data_index])-100):
                ts = self.files_dict[data_index][i]
                if float(ts)-float(file_name) > self.max_t:
                    break
                else:
                    x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                    x_list.append(x_)
                    y_list.append(y_)
                    vx_ = self.vel_dict[data_index][ts][0]
                    vy_ = self.vel_dict[data_index][ts][1]
                    vx = np.cos(yaw)*vx_ + np.sin(yaw)*vy_
                    vy = np.cos(yaw)*vy_ - np.sin(yaw)*vx_
                    
                    ax_ = self.acc_dict[data_index][ts][0]
                    ay_ = self.acc_dict[data_index][ts][1]
                    ax = ax_*np.cos(yaw) + ay_*np.sin(yaw)
                    ay = ay_*np.cos(yaw) - ax_*np.sin(yaw)
                    theta_a = np.arctan2(ay, ax)
                    theta_v = np.arctan2(vy, vx)
                    sign = np.sign(np.cos(theta_a-theta_v))
                    a = sign*np.sqrt(ax*ax + ay*ay)
                    a_list.append(a)
                    vx_list.append(vx)
                    vy_list.append(vy)
                    ts_list.append(ts)
                    relative_t_list.append(float(ts)-float(file_name))
                        
            if len(ts_list) == 0:
                continue
            else:
                #ts = random.sample(ts_list, 1)[0]
                ts_array = random.sample(ts_list, self.traj_steps)
                #weights = [np.exp(-0.23*(float(ts)-float(file_name))) for ts in ts_list]
                #sample_ts = random.choices(ts_list, weights)[0]
                #print(weights/sum(weights))
                break
            
        #ts = sample_ts
        # [0 ~ 1]
        # v0
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        v_0 = np.sqrt(_vx_0*_vx_0 + _vy_0*_vy_0)
        v_0 = torch.FloatTensor([v_0])
        v0_array = [v_0]*self.traj_steps

        t_array = []
        xy_array = []
        vxy_array = []
        axy_array = []
        a_array = []
        for ts in ts_array:
            t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
            t_array.append(t)
            # x, y
            x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
            xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])# [-1, 1]
            xy_array.append(xy)
            # yaw_t
            #yaw_t = angle_normal(np.deg2rad(self.pose_dict[data_index][ts][3]) - yaw)
            #yaw_t = torch.FloatTensor([yaw_t/np.pi])# [-1, 1]
            
            # vx, vy
            _vx = self.vel_dict[data_index][ts][0]
            _vy = self.vel_dict[data_index][ts][1]
            vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
            vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
            vxy_array.append(torch.FloatTensor([vx, vy]))
            
            # ax, ay
            _ax = self.acc_dict[data_index][ts][0]
            _ay = self.acc_dict[data_index][ts][1]
            ax = _ax*np.cos(yaw) + _ay*np.sin(yaw)
            ay = _ay*np.cos(yaw) - _ax*np.sin(yaw)
            axy_array.append(torch.FloatTensor([ax, ay]))
            
            theta_a = np.arctan2(_ay, _ax)
            theta_v = np.arctan2(_vy, _vx)
            sign = np.sign(np.cos(theta_a-theta_v))
            a = sign*np.sqrt(ax*ax + ay*ay)
            a_array.append(a)
        
        t = torch.FloatTensor(t_array)
        v_0 = torch.FloatTensor(v0_array)
        xy = torch.stack(xy_array)
        vxy = torch.stack(vxy_array)
        axy = torch.stack(axy_array)
        a = torch.FloatTensor(a_array)
        
        #vxy = torch.FloatTensor([vx, vy])
        #axy = torch.FloatTensor([ax, ay])
        x_list = torch.FloatTensor(x_list)
        y_list = torch.FloatTensor(y_list)
        vx_list = torch.FloatTensor(vx_list)
        vy_list = torch.FloatTensor(vy_list)
        a_list = torch.FloatTensor(a_list)
        relative_t_list = torch.FloatTensor(relative_t_list)
        
        if self.evalmode:
            return {'img': imgs, 't': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'a':a, 'v_0':v_0,
                    'a_list':a_list,
                    'x_list':x_list, 'y_list':y_list,
                    'vx_list':vx_list, 'vy_list':vy_list,
                    'ts_list':relative_t_list}
        else:
            return {'img': imgs, 't': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'a':a, 'v_0':v_0}
"""