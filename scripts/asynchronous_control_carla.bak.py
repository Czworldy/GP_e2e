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
from learning.models import GeneratorUNet
from learning.path_model import ModelGRU

from ff_collect_pm_data import sensor_dict
from utils.collect_ipm import InversePerspectiveMapping
from utils.carla_sensor import Sensor, CarlaSensorMaster
from utils.capac_controller import CapacController
#from utils.train_utils import get_diff_tf

from utils.gym_wrapper import CARLAEnv
from utils.pre_process import generate_costmap, get_costmap_stack
from utils.post_process import draw_traj, visualize
from utils import GlobalDict
import carla_utils as cu

import os
import cv2
import time
import copy
import threading
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch.autograd import grad
import torchvision.transforms as transforms

import gym

global_dict = GlobalDict()

global_dict['img'] = None
global_dict['view_img'] = None
global_dict['pcd'] = None
global_dict['nav'] = None
global_dict['cost_map'] = None

global_dict['vel'] = 0.
global_dict['a'] = 0.
global_dict['cnt'] = 0
global_dict['v0'] = 0.
global_dict['plan_time'] = 0.
global_dict['trajectory'] = None
global_dict['vehicle'] = None
global_dict['plan_map'] = None
global_dict['transform'] = None
global_dict['draw_cost_map'] = None
global_dict['max_steer_angle'] = 0.
global_dict['ipm_image'] = np.zeros((200,400), dtype=np.uint8)
global_dict['ipm_image'].fill(255)
global_dict['trans_costmap_dict'] = {}
global_dict['state0'] = None
global_dict['start_control'] = False

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = GeneratorUNet()
generator = generator.to(device)
generator.load_state_dict(torch.load('../ckpt/g.pth'))
model = ModelGRU().to(device)
model.load_state_dict(torch.load('../ckpt/gru.pth'))
generator.eval()
model.eval()

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-s', '--save', type=bool, default=False, help='save result')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--trans_width', type=int, default=256, help='transform image width')
parser.add_argument('--trans_height', type=int, default=128, help='transform image height')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--dt', type=float, default=0.03, help='discretization minimum time interval')
parser.add_argument('--max_speed', type=float, default=30., help='max speed')
parser.add_argument('--rnn_steps', type=int, default=10, help='rnn readout steps')

args = parser.parse_args()

data_index = args.data
save_path = '/media/wang/DATASET/CARLA/town01/'+str(data_index)+'/'

img_transforms = [
    transforms.Resize((args.trans_height, args.trans_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)

cost_map_transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ]
cost_map_trans = transforms.Compose(cost_map_transforms_)
        
class Param(object):
    def __init__(self):
        self.longitudinal_length = args.scale
        self.ksize = 21
        
param = Param()
sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)

def mkdir(path):
    os.makedirs(save_path+path, exist_ok=True)

def image_callback(data):
    global_dict['plan_time'] = time.time()
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_dict['img'] = array
    global_dict['transform'] = global_dict['vehicle'].get_transform()
    try:
        global_dict['nav'] = get_nav(global_dict['vehicle'], global_dict['plan_map'])
        v = global_dict['vehicle'].get_velocity()
        global_dict['v0'] = np.sqrt(v.x**2+v.y**2+v.z**2)
    except:
        pass
    
def view_image_callback(data):
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_dict['view_img'] = array
    
def lidar_callback(data):
    ts = time.time()
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where(point_cloud[2] > -2.3)[0]
    point_cloud = point_cloud[:, mask]
    global_dict['pcd'] = point_cloud
    generate_costmap(ts, global_dict, cost_map_trans, args)

def get_cGAN_result(img, nav):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(nav,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0).to(device)
    result = generator(input_img)
    result = result.cpu().data.numpy()[0][0]*127+128
    result = cv2.resize(result, (global_dict['img'].shape[1], global_dict['img'].shape[0]), interpolation=cv2.INTER_CUBIC)
    return result
    
def get_traj(plan_time):
    global_dict['state0'] = cu.getActorState('odom', plan_time, global_dict['vehicle'])
    global_dict['state0'].x = global_dict['transform'].location.x
    global_dict['state0'].y = global_dict['transform'].location.y
    global_dict['state0'].z = global_dict['transform'].location.z
    global_dict['state0'].theta = np.deg2rad(global_dict['transform'].rotation.yaw)
    
    t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
    t.requires_grad = True

    trans_costmap_stack = get_costmap_stack(global_dict)
    trans_img = torch.stack(trans_costmap_stack)
    img = trans_img.expand(len(t),trans_img.shape[0],1,args.height, args.width)
    # for resnet backbone
    #img = trans_img.expand(len(t),3,args.height, args.width)
    img = img.to(device)
    img.requires_grad = True
    v_0 = torch.FloatTensor([global_dict['v0']]).expand(len(t),1)
    v_0 = v_0.to(device)

    output = model(img, t, v_0)
    vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(args.max_dist/args.max_t)
    vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(args.max_dist/args.max_t)
    
    ax = grad(vx.sum(), t, create_graph=True)[0][:,0]/args.max_t
    ay = grad(vy.sum(), t, create_graph=True)[0][:,0]/args.max_t

    output_axy = torch.cat([ax.unsqueeze(1), ay.unsqueeze(1)], dim=1)

    x = output[:,0]*args.max_dist
    y = output[:,1]*args.max_dist
    
    theta_a = torch.atan2(ay, ax)
    theta_v = torch.atan2(vy, vx)
    sign = torch.sign(torch.cos(theta_a-theta_v))
    a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)
    
    # draw
    global_dict['draw_cost_map'] = draw_traj(global_dict['cost_map'], output, args)
    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    ax = ax.data.cpu().numpy()
    ay = ay.data.cpu().numpy()
    a = a.data.cpu().numpy()

    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
    return trajectory

def make_plan():
    while True:
        #t1 = time.time()
        plan_time = global_dict['plan_time']
        # 1. get cGAN result
        result = get_cGAN_result(global_dict['img'], global_dict['nav'])
        # 2. inverse perspective mapping and get costmap
        img = copy.deepcopy(global_dict['img'])
        mask = np.where(result > 200)
        img[mask[0],mask[1]] = (255, 0, 0, 255)
        
        ipm_image = inverse_perspective_mapping.getIPM(result)
        global_dict['ipm_image'] = ipm_image

        # 3. get trajectory
        #time.sleep(0.1)
        global_dict['trajectory'] = get_traj(plan_time)

        if not global_dict['start_control']:
            global_dict['start_control'] = True
        #t2 = time.time()
        #print('time:', 1000*(t2-t1))
    
def main():
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
    #vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.yamaha.yzf')
    #vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.*.*')
    global_dict['vehicle'] = vehicle
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()
    global_dict['max_steer_angle'] = np.deg2rad(physics_control.wheels[0].max_steer_angle)
    sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':image_callback,
            },
        'camera:view':{
            #'transform':carla.Transform(carla.Location(x=0.0, y=4.0, z=4.0), carla.Rotation(pitch=-30, yaw=-60)),
            'transform':carla.Transform(carla.Location(x=-3.0, y=0.0, z=6.0), carla.Rotation(pitch=-45)),
            #'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=6.0), carla.Rotation(pitch=-90)),
            'callback':view_image_callback,
            },
        'lidar':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':lidar_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    
    #spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    agent = BasicAgent(vehicle, target_speed=args.max_speed)
    
    # prepare map
    destination = carla.Transform()
    destination.location = world.get_random_location_from_navigation()
    #destination = get_random_destination(spawn_points)
    global_dict['plan_map'] = replan(agent, destination, copy.deepcopy(origin_map))

    # start to plan
    plan_thread = threading.Thread(target = make_plan, args=())
    #visualization_thread = threading.Thread(target = show_traj, args=())

    while True:
        if (global_dict['img'] is not None) and (global_dict['nav'] is not None) and (global_dict['pcd'] is not None):
            plan_thread.start()
            break
        else:
            time.sleep(0.001)
    
    # wait for the first plan result
    while not global_dict['start_control']:
        time.sleep(0.001)
    
    #visualization_thread.start()
    # start to control
    print('Start to control')
    
    ctrller = CapacController(world, vehicle, 30)
    
    env = CARLAEnv(world, vehicle)
    env.reset()
    #env.step()
    
    while True:
        # change destination
        if close2dest(vehicle, destination):
            #destination = get_random_destination(spawn_points)
            print('get destination !', time.time())
            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()
            global_dict['plan_map'] = replan(agent, destination, copy.deepcopy(origin_map)) 

        v = global_dict['vehicle'].get_velocity()
        a = global_dict['vehicle'].get_acceleration()
        global_dict['vel'] = np.sqrt(v.x**2+v.y**2+v.z**2)
        global_dict['a'] = np.sqrt(a.x**2+a.y**2+a.z**2)
        #steer_angle = global_dict['vehicle'].get_control().steer*global_dict['max_steer_angle']
        #w = global_vel*np.tan(steer_angle)/2.405
        
        control_time = time.time()
        dt = control_time - global_dict['trajectory']['time']
        index = int((dt/args.max_t)//args.dt) + 5
        if index > 0.99/args.dt:
            continue
        
        """
        transform = vehicle.get_transform()
        
        dx, dy, dyaw = get_diff_tf(transform, global_dict['transform'])
        dyaw = -dyaw
        
        _x = global_dict['trajectory']['x'][index] - dx
        _y = global_dict['trajectory']['y'][index] - dy
        x = _x*np.cos(dyaw) + _y*np.sin(dyaw)
        y = _y*np.cos(dyaw) - _x*np.sin(dyaw)
        
        _vx = global_dict['trajectory']['vx'][index]
        _vy = global_dict['trajectory']['vy'][index]
        vx = _vx*np.cos(dyaw) + _vy*np.sin(dyaw)
        vy = _vy*np.cos(dyaw) - _vx*np.sin(dyaw)
        
        _ax = global_dict['trajectory']['ax'][index]
        _ay = global_dict['trajectory']['ay'][index]
        ax = _ax*np.cos(dyaw) + _ay*np.sin(dyaw)
        ay = _ay*np.cos(dyaw) - _ax*np.sin(dyaw)
        """
        control = ctrller.run_step(global_dict['trajectory'], index, global_dict['state0'])
        env.step(control)
        #vehicle.apply_control(control)
        """
        dyaw = np.deg2rad(global_dict['transform'].rotation.yaw)
        x = global_dict['trajectory']['x'][index]*np.cos(dyaw) + global_dict['trajectory']['y'][index]*np.sin(dyaw)
        y = global_dict['trajectory']['y'][index]*np.cos(dyaw) - global_dict['trajectory']['x'][index]*np.sin(dyaw)
        """ 
        curve = None#show_traj(True)
        #visualize(global_dict['view_img'], global_dict['draw_cost_map'], global_dict['nav'], args, curve)
        visualize(global_dict, global_dict['draw_cost_map'], args, curve)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()