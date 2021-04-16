#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla
sys.path.append('/home/wang/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_map, get_nav, replan, close2dest
from utils.train_utils import get_transform

import cv2
import time
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import gym
from gym import spaces
from gym.utils import seeding

import torch
from torch.autograd import grad
import torchvision.transforms as transforms

from model import Generator, Discriminator

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--longitudinal_length', type=float, default=25., help='image height')
parser.add_argument('--vector_dim', type=int, default=2, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
args = parser.parse_args()

generator = Generator(args.vector_dim+2).to(device)
discriminator = Discriminator(args.points_num*2+1).to(device)

#discriminator.load_state_dict(torch.load('result/saved_models/wgan-gp-10/discriminator_40000.pth'))
#generator.load_state_dict(torch.load('result/saved_models/wgan-gp-10/generator_40000.pth'))

global_img = None
global_view_img = None
global_pcd = None

def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array
    
def view_image_callback(data):
    global global_view_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_view_img = array
    
def lidar_callback(data):
    global global_pcd
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where(point_cloud[2] > -2.3)[0]
    point_cloud = point_cloud[:, mask]
    global_pcd = point_cloud

def get_cost_map(img, point_cloud):
    img2 = np.zeros((args.height, args.width), np.uint8)
    img2.fill(255)
    
    pixs_per_meter = args.height/args.longitudinal_length
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

def xy2uv(x, y):
    pixs_per_meter = args.height/args.longitudinal_length
    u = (args.height-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+args.width//2).astype(int)
    return u, v

class CARLAEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        # time step
        self.dt = 1.0 / 30
        # robot ability
        self.max_v = 3.0
        self.max_w = 2*np.pi
        self.robot_max_acc = 3
        self.robot_max_w_acc = 2*np.pi
        # robot ability for each step
        self.max_step_acc = self.robot_max_acc * self.dt
        self.max_step_w_acc = self.robot_max_w_acc * self.dt
        # robot action space
        self.low_action = np.array([-1, -1])
        self.high_action = np.array([1, 1])
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)
        # robot observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # environment feedback infomation
        self.state = []
        self.done = False
        self.reward = 0.0
        
        self.seed()
        self.reset()

    def step(self, action=[0,0]):
        
        #total reward
        self.reward = 0
        self.state = []

        return self.state, self.reward, self.done, {}

    def reset(self):

        self.state = []
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

def main():
    global global_img, global_view_img, global_pcd
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.load_world('Town01')

    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,10),
        precipitation=0,
        sun_altitude_angle=random.randint(50,90)
    )
    
    set_weather(world, weather)
    
    blueprint = world.get_blueprint_library()
    world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')

    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()
    #max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)
    sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':image_callback,
            },
        'camera:view':{
            'transform':carla.Transform(carla.Location(x=-3.0, y=0.0, z=8.0), carla.Rotation(pitch=-45)),
            'callback':view_image_callback,
            },
        'lidar':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':lidar_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()

    #agent = BasicAgent(vehicle, target_speed=MAX_SPEED)

    destination = carla.Transform()
    destination.location = world.get_random_location_from_navigation()

    env = CARLAEnv()
    env.reset()
    env.step()

    # start to control
    print('Start to control')
    vehicle.set_autopilot(True)
    while True:
        # change destination
        if close2dest(vehicle, destination):
            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()

        #control = None
        #vehicle.apply_control(control)
        #localtion = carla.Location(x = global_transform.location.x+x, y=global_transform.location.y+y, z=2.0)
        #world.debug.draw_point(localtion, size=0.3, color=carla.Color(255,0,0), life_time=10.0)

        forward_vector = vehicle.get_transform().get_forward_vector()
        begin = vehicle.get_location()
        end = vehicle.get_location()
        end.x = end.x + forward_vector.x*8
        end.y = end.y + forward_vector.y*8
        end.z = end.z + forward_vector.z*8
        world.debug.draw_line(begin, end, thickness=0.2, 
            color=carla.Color(r=255,g=0,b=0,a=128),
            life_time=0.1)

        cv2.imshow('Visualization', global_view_img)
        cv2.waitKey(5)

        #time.sleep(1/60.)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()