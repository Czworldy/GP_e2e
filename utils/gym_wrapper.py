#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import random
import copy
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from torch.autograd import grad
from .pre_process import get_costmap_stack
from .capac_controller import CapacController
from .post_process import visualize, draw_traj
from .navigator_sim import get_map, replan, close2dest

import simulator
simulator.load('/home/cz/CARLA_0.9.9.4')
import carla
sys.path.append('/home/cz/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CARLAEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, world, vehicle, global_dict, args, trajectory_model):
        self.world = world
        self.vehicle = vehicle
        self.agent = BasicAgent(self.vehicle, target_speed=30)
        self.global_dict = global_dict
        self.args = args
        self.trajectory_model = trajectory_model
        self.ctrller = CapacController(self.world, self.vehicle, 50)
        # robot action space
        self.low_action = np.array([-1, -1])
        self.high_action = np.array([1, 1])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)
        # robot observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        world_map = self.world.get_map()
        waypoint_tuple_list = world_map.get_topology()
        self.origin_map = get_map(waypoint_tuple_list)
        self.spawn_points = world_map.get_spawn_points()
        self.route_trace = None
        # environment feedback infomation
        self.state = []
        self.done = False
        self.reward = 0.0
        
        self.seed()
        self.reset()

    #def step(self, action=[0,0]):
    def step(self, action, plan_time):

        t = torch.arange(0, 0.99, self.args.dt).unsqueeze(1).to(device)
        t.requires_grad = True
        self.global_dict['v0'] = max(3.0, self.global_dict['v0'])
        v_0 = torch.FloatTensor([self.global_dict['v0']/self.args.max_speed]).expand(len(t),1)
        v_0 = v_0.to(device)
        # t_with_v = torch.cat([t, v_0], dim=1)
        
        noise = torch.FloatTensor(action).unsqueeze(0).unsqueeze(0)
        noise = noise.expand(1, len(t), self.args.vector_dim)
        noise = noise.reshape(1*len(t), self.args.vector_dim)
        #noise.requires_grad = True
        noise = noise.to(device)

        # print("noise shape:", noise.shape)
        # print("t_with_v shape:", t_with_v.shape)
        # print("t shape:", t.shape)
        # print("v0 shape:", v_0.shape)
        # output_xy = self.trajectory_model(noise, t_with_v, t)
        output_xy = self.trajectory_model(v_0, noise, t)

        #print("output_xy:", output_xy.shape)
        # print(output_xy[:,0].sum())
        vx = grad(output_xy[:,0].sum(), t, create_graph=True)[0][:,0]*(self.args.max_dist/self.args.max_t)
        vy = grad(output_xy[:,1].sum(), t, create_graph=True)[0][:,0]*(self.args.max_dist/self.args.max_t)
        
        ax = grad(vx.sum(), t, create_graph=True)[0][:,0]/self.args.max_t
        ay = grad(vy.sum(), t, create_graph=True)[0][:,0]/self.args.max_t
    
        output_axy = torch.cat([ax.unsqueeze(1), ay.unsqueeze(1)], dim=1)
    
        x = output_xy[:,0]*self.args.max_dist
        y = output_xy[:,1]*self.args.max_dist
        
        theta_a = torch.atan2(ay, ax)
        theta_v = torch.atan2(vy, vx)
        sign = torch.sign(torch.cos(theta_a-theta_v))
        a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)
        
        # draw
        #self.global_dict['draw_cost_map'] = draw_traj(self.global_dict['cost_map'], output_xy, self.args)
        
        vx = vx.data.cpu().numpy()
        vy = vy.data.cpu().numpy()
        x = x.data.cpu().numpy()
        y = y.data.cpu().numpy()
        ax = ax.data.cpu().numpy()
        ay = ay.data.cpu().numpy()
        a = a.data.cpu().numpy()
    
        trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
        
        self.reward = 0.
        waypoint1, index1, diff_deg1 = self.find_waypoint()
        waypoint = carla.Location(x=waypoint1.location.x, y=waypoint1.location.y, z=2.0)
        self.world.debug.draw_point(waypoint, size=0.2, color=carla.Color(255,0,0), life_time=1.0)
        
        old_loc_x = self.vehicle.get_transform().location.x
        old_loc_y = self.vehicle.get_transform().location.y
        org_dist = np.sqrt((self.vehicle.get_transform().location.x-waypoint1.location.x)**2+(self.vehicle.get_transform().location.y-waypoint1.location.y)**2)
        vehicle2point = np.arctan2(waypoint1.location.y-self.vehicle.get_transform().location.y, waypoint1.location.x-self.vehicle.get_transform().location.x)
        vehicle_yaw = np.deg2rad(self.vehicle.get_transform().rotation.yaw)
        delta_theta = abs(self._angle_normalize(vehicle2point-vehicle_yaw))
        # error = -org_dist * np.sin(delta_theta)
        """
        for i in range(index1, min(index1+10, len(self.route_trace)-1)):
            location = self.route_trace[i][0].transform.location
            x = location.x
            y = location.y
            waypoint = carla.Location(x=x, y=y, z=2.0)
            self.world.debug.draw_point(waypoint, size=0.2, color=carla.Color(255,0,0), life_time=0.5)
        """
        for i in range(50):
            if self.global_dict['collision']:
                self.done = True
                self.reward -= 50.  #-50
                print('Done !')
                break
            if close2dest(self.vehicle, self.destination):
                self.done = True
                self.reward += 100.
                print('Success !')
                break
            
            """
            for i in range(len(self.route_trace)):
                x = self.route_trace[i][0].transform.location.x
                y = self.route_trace[i][0].transform.location.y
                localtion = carla.Location(x=x, y=y, z=2.0)
                self.world.debug.draw_point(localtion, size=0.2, color=carla.Color(255,0,0), life_time=0.5)
            """
            if self.args.show and i % 5 == 0:
                self.global_dict['draw_cost_map'] = draw_traj(self.global_dict['cost_map'], output_xy, self.args)
                visualize(self.global_dict, self.global_dict['draw_cost_map'], self.args, curve=None)
            control_time = time.time()
            dt = control_time - trajectory['time']
            index = int((dt/self.args.max_t)//self.args.dt) + 8
            if index > 0.99//self.args.dt-10:
                continue
            
            control = self.ctrller.run_step(trajectory, index, self.global_dict['state0'])
            self.vehicle.apply_control(control)
            time.sleep(0.01)
            
        """
        trans_costmap_stack = get_costmap_stack(self.global_dict)
        trans_costmap_stack = torch.stack(trans_costmap_stack)
        self.state = trans_costmap_stack
        """
        new_dist = np.sqrt((self.vehicle.get_transform().location.x-waypoint1.location.x)**2+(self.vehicle.get_transform().location.y-waypoint1.location.y)**2)
        vehicle2point_new = np.arctan2(waypoint1.location.y-self.vehicle.get_transform().location.y, waypoint1.location.x-self.vehicle.get_transform().location.x)
        vehicle_yaw_new = np.deg2rad(self.vehicle.get_transform().rotation.yaw)
        delta_thet_new = abs(self._angle_normalize(vehicle2point_new-vehicle_yaw_new))
        
        # error_new = -new_dist * np.sin(delta_thet_new)
        dist_reward = (org_dist-new_dist)#/10.0
        # theta_reward = (delta_theta - delta_thet_new) # * 60 add by yujiyu 
        # theta_reward = (error - error_new)*100
        step_dist = np.sqrt((self.vehicle.get_transform().location.x - old_loc_x)**2+(self.vehicle.get_transform().location.y - old_loc_y)**2)

        diff_deg1 = abs(np.rad2deg(self._angle_normalize(np.deg2rad(diff_deg1))))
        if diff_deg1 < 10:
            theta_reward = (10 - diff_deg1)/3
        else:
            theta_reward = - diff_deg1 / 10.
        theta_reward = np.clip(theta_reward, -3, 3)
        # self.reward += dist_reward
        self.reward += theta_reward
        self.reward += step_dist * 2.
        v = self.vehicle.get_velocity()
        v0 = np.sqrt(v.x**2+v.y**2+v.z**2) / 2. #/4
        self.reward += v0
        print( "v:%.2f theta:%.2f dist:%.2f " % (v0, theta_reward, step_dist) )
        self.state = copy.deepcopy(self.global_dict['trans_costmap'])
        #self.global_dict['ts']
        
        #waypoint2, index2, diff_deg2 = self.find_waypoint()
        #print(index2, index1)

        return self.state, self.reward, self.done, copy.deepcopy(self.global_dict['ts'])

    def find_waypoint(self):
        position = self.vehicle.get_transform().location
        yaw = self.vehicle.get_transform().rotation.yaw
        asb_dists = []
        for i in range(len(self.route_trace)):
            location = self.route_trace[i][0].transform.location
            x = location.x
            y = location.y
            asb_dists.append(np.sqrt((x-position.x)**2+(y-position.y)**2))
            #waypoint = carla.Location(x=x, y=y, z=2.0)
            #self.world.debug.draw_point(waypoint, size=0.2, color=carla.Color(255,0,0), life_time=0.5)
        min_val = min(asb_dists)
        index = asb_dists.index(min_val)
        #if min_val < 1.0:
        index += 5
        index = np.clip(index, 0, len(self.route_trace)-1)
        try:
            x0 = self.route_trace[index][0].transform.location.x
            y0 = self.route_trace[index][0].transform.location.y
            x1 = self.route_trace[index+1][0].transform.location.x
            y1 = self.route_trace[index+1][0].transform.location.y
            wp_yaw = np.arctan2(y1-y0, x1-x0)
            #wp_yaw = np.rad2deg(wp_yaw)

        except:
            wp_yaw = 0.0
        #print('11111', self.route_trace[index][0].transform.rotation.yaw, yaw)
        #print('vehicle:', yaw)
        yaw = np.deg2rad(yaw)
        #print('ddddddddd', np.rad2deg(abs(wp_yaw-yaw)))
        #print(wp_yaw, yaw)
        return self.route_trace[index][0].transform, index, np.rad2deg(abs(wp_yaw-yaw))
        
    def reset(self):
        # start_point = random.choice(self.spawn_points)
        # yujiyu
        start_point = self.spawn_points[2]
        self.destination = random.choice(self.spawn_points)
        
        self.vehicle.set_transform(start_point)
        
        self.global_dict['plan_map'], self.destination = replan(self.agent, self.destination, copy.deepcopy(self.origin_map), self.spawn_points)
        
        self.global_dict['collision'] = False
        
        start_waypoint = self.agent._map.get_waypoint(self.agent._vehicle.get_location())
        end_waypoint = self.agent._map.get_waypoint(self.destination.location)

        self.route_trace = self.agent._trace_route(start_waypoint, end_waypoint)
        start_point.rotation = self.route_trace[0][0].transform.rotation
        self.vehicle.set_transform(start_point)

        #yujiyu
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        #
        """
        trans_costmap_stack = get_costmap_stack(self.global_dict)
        trans_costmap_stack = torch.stack(trans_costmap_stack)
        self.state = trans_costmap_stack[0]
        """
        self.state = self.global_dict['trans_costmap']
        self.done = False
        self.reward = 0.0
        # print('RESET !!!!!!!!')
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _angle_normalize(self, angle):
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi
        return angle
