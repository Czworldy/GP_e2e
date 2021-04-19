#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import random
import copy
import gym
import cv2
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from torch.autograd import grad
from .capac_controller import CapacController
# from .post_process import visualize, draw_traj
from .navigator_sim_route import get_map, replan, close2dest
from .route import get_reference_route
from matplotlib import pyplot as plt 
import torchvision.transforms as transforms

import simulator
simulator.load('/home/cz/CARLA_0.9.9.4')
import carla
sys.path.append('/home/cz/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent
from .pid import LongPID


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_height = 200
img_width = 400
state_trans_0 = [
    transforms.Resize((img_height, img_width)),
    transforms.ToPILImage(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
state_trans = transforms.Compose(state_trans_0)
def get_junctions(waypoint_pairs):
    waypoint_ids = []
    waypoints = []
    for wp_pair in waypoint_pairs:
        for wp in wp_pair:
            if wp.id not in waypoint_ids:
                waypoint_ids.append(wp.id)
                waypoints.append(wp)
    return waypoints

def visualize(img, nav, speed, imput):
    text = "speed: "+str(round(3.6*speed, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)    
    cv2.imshow('Visualization', img)
    cv2.imshow('Nav', nav)
    # cv2.imshow('input', imput)
    cv2.waitKey(1)

# def draw_traj(nav_map, output, args):
#     nav_map = Image.fromarray(nav_map).convert("RGB")
#     draw = ImageDraw.Draw(nav_map)
#     result = output.data.cpu().numpy()
#     x = args.max_dist*result[:,0]
#     y = args.max_dist*result[:,1]
#     u, v = xy2uv(x, y, args)
#     for i in range(len(u)-1):
#         draw.line((v[i], u[i], v[i+1], u[i+1]), 'red')
#         draw.line((v[i]+1, u[i], v[i+1]+1, u[i+1]), 'red')
#         draw.line((v[i]-1, u[i], v[i+1]-1, u[i+1]), 'red')
#     return nav_map
    
class CARLAEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, world, vehicle, global_dict, args):
        self.world = world
        self.vehicle = vehicle
        self.agent = BasicAgent(self.vehicle, target_speed=40)
        self.global_dict = global_dict
        self.args = args
        self.pid = LongPID(0.05, -0.5, 1.0)
        # self.ctrller = CapacController(self.world, self.vehicle, 30) #freq=50
        # robot action space

        # self.low_action = np.array([-1, -1])
        # self.high_action = np.array([1, 1])
        # self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        # robot observation space

        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.world_map = self.world.get_map()
        self.waypoint_tuple_list = self.world_map.get_topology()
        self.origin_map = get_map(self.waypoint_tuple_list)
        self.spawn_points = self.world_map.get_spawn_points()
        self.route_trace = None
        # environment feedback infomation
        self.state = {}
        self.done = False
        self.reward = 0.0
        
        self.seed()
        self.reset()


    def step(self, action):

        # throttle = action[0].astype("float64")
        steer = action[0].astype("float64")

        self.reward = 0.
        """oringal
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
        for _ in range(4): #50

            if self.global_dict['collision']:
                self.done = True
                self.reward -= 1.  #-50
                print('collision !')
                break
            if close2dest(self.vehicle, self.destination, dist=5):
                self.done = True
                self.reward += 10.   # reward += 100
                print('Success !')
                break
            current_speed_carla = self.vehicle.get_velocity()
            current_speed_kmh = np.sqrt(current_speed_carla.x**2+current_speed_carla.y**2) * 3.6
            throttle, brake = self.pid.run_step(current_speed_kmh, 16.)
            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            self.vehicle.apply_control(control)
            self.world.tick()
            # visualize(self.global_dict['view_img'], self.global_dict['nav'], self.global_dict['v0'], self.global_dict['img'] )
            # time.sleep(0.01)

        waypoint1, _, diff_rad = self.find_waypoint()
        vehicle_current_x = self.vehicle.get_transform().location.x
        vehicle_current_y = self.vehicle.get_transform().location.y
        # vehicle_current_yaw = self.vehicle.get_transform().rotation.yaw
        v = self.vehicle.get_velocity()
        # waypoint = self.world_map.get_waypoint(self.vehicle.get_transform().location)

        trace_dist = np.sqrt((waypoint1.location.x - vehicle_current_x) ** 2 + (waypoint1.location.y - vehicle_current_y) ** 2)
        
        if waypoint1 is not None:
            lane_x = waypoint1.location.x
            lane_y = waypoint1.location.y
            lane_offset = np.sqrt( (lane_x - vehicle_current_x) ** 2 + (lane_y - vehicle_current_y) ** 2) 
            # print("lane_offset: %.2f"  %(lane_offset))
            # if lane_offset > 0.7 and lane_offset <= 1.8:  #转弯的时候还是存在问题 route.py 解决
            if lane_offset <= 3.0:
                lane_reward = 1 - lane_offset
                
            # elif lane_offset > 1.5 and lane_offset <= 3.3:
            #     lane_reward = (0.7 - lane_offset)
            elif lane_offset > 3.0:
                print('off line!')
                self.done = True
                lane_reward = 0
                self.reward -= 4
            else:
                lane_reward = 0
                
        else:
            print('waypoint not found!')
            lane_reward = -4
        yaw_reward = -3 * diff_rad/ np.pi
        self.reward += yaw_reward
        self.reward += lane_reward
        
        kmh = np.sqrt(v.x**2+v.y**2) * 3.6
        # if kmh < 10:
        #     kmh_reward = (kmh - 10) / 10.
        # else:
        #     kmh_reward = (kmh - 10) / 10.
        kmh_reward = (kmh - 10) / 10.
        self.reward += kmh_reward

        """
        new_dist = np.sqrt((vehicle_current_x-waypoint1.location.x)**2+(vehicle_current_y-waypoint1.location.y)**2)
        vehicle2point_new = np.arctan2(waypoint1.location.y - vehicle_current_y, waypoint1.location.x - vehicle_current_x)
        vehicle_yaw_new = np.deg2rad(vehicle_current_yaw)
        delta_thet_new = abs(self._angle_normalize(vehicle2point_new-vehicle_yaw_new))
        
        # error_new = -new_dist * np.sin(delta_thet_new)
        dist_reward = (org_dist-new_dist)/10.
        theta_reward = (delta_theta - delta_thet_new) 
        # theta_reward = (error - error_new)*100
        # step_dist = np.sqrt((self.vehicle.get_transform().location.x - old_loc_x)**2+(self.vehicle.get_transform().location.y - old_loc_y)**2)


        self.reward += dist_reward
        self.reward += theta_reward
        

        v0 = np.sqrt(v.x**2+v.y**2+v.z**2) / 4. #/4
        self.reward += v0
        """


        print( "reward: %.2f , lane_reward: %.2f , trace_dist: %.2f , yaw_reward:%.2f , kmh_reward:%.2f" % (self.reward, lane_reward, trace_dist, yaw_reward,kmh_reward) )
        # print("trace_dist: %.2f" % (trace_dist))
        self.state['img_nav'] = copy.deepcopy(self.global_dict['img_nav'])

        
        #self.global_dict['ts']
        # state_save = state_trans(self.state[:3])
        # state_save = np.array(state_save)
        # print(state_save.shape, np.max(state_save), np.min(state_save))
        # cv2.imwrite('/home/cz/result/img/state{}.png'.format(str(time.time())), state_save)
        #waypoint2, index2, diff_deg2 = self.find_waypoint()
        #print(index2, index1)

        return self.state, self.reward, self.done, self.global_dict['ts']

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
        # index += 5
        # index = np.clip(index, 0, len(self.route_trace)-1)
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
        err = abs(wp_yaw-yaw)
        err = abs(self._angle_normalize(err))
        #print('ddddddddd', np.rad2deg(abs(wp_yaw-yaw)))
        #print(wp_yaw, yaw)
        return self.route_trace[index][0].transform, index, err
        
    def reset(self):
        # start_point = random.choice(self.spawn_points)
        # self.destination = random.choice(self.spawn_points)
        # yujiyu
        self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        junctions = get_junctions(self.waypoint_tuple_list)

        start_point = random.choice(junctions).transform
        # start_point = self.spawn_points[1]  #1
        # self.vehicle.set_transform(start_point)
        
        for _ in range(10):
            self.world.tick()

        ref_route = get_reference_route(self.world_map, start_point.location, 50, 0.05)
        self.destination = ref_route[-1][0].transform
        
        self.global_dict['plan_map'], self.destination, ref_route, start_point= replan(self.world_map, self.vehicle, self.agent, self.destination, copy.deepcopy(self.origin_map), self.spawn_points)
        
        # show_plan = cv2.cvtColor(np.asarray(self.global_dict['plan_map']), cv2.COLOR_BGR2RGB)
        # cv2.namedWindow('plan_map', 0)    
        # cv2.resizeWindow('plan_map', 600, 600)   # 自己设定窗口图片的大小
        # cv2.imshow('plan_map', show_plan)
        # cv2.waitKey(1)

        self.global_dict['collision'] = False
        
        # start_waypoint = self.agent._map.get_waypoint(self.agent._vehicle.get_location())
        # end_waypoint = self.agent._map.get_waypoint(self.destination.location)

        # self.route_trace = self.agent._trace_route(start_waypoint, end_waypoint)
        self.route_trace = ref_route
        start_point.rotation = self.route_trace[0][0].transform.rotation
        self.vehicle.set_transform(start_point)
        for _ in range(10):
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
            self.world.tick()

        self.state['img_nav'] = copy.deepcopy(self.global_dict['img_nav'])

        # if self.state == None:
        #     print("None State!")

        self.done = False
        self.reward = 0.0
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
