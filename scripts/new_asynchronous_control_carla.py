#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../'))

import simulator
simulator.load('/home/cz/CARLA_0.9.9.4')
import carla
sys.path.append('/home/cz/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_map, get_nav, replan, close2dest
# from learning.models import GeneratorUNet
# from learning.path_model import ModelGRU
# from utils import fig2data, add_alpha_channel

from ff_collect_pm_data import sensor_dict
# from utils.collect_ipm import InversePerspectiveMapping
from utils.carla_sensor import Sensor, CarlaSensorMaster
from utils.capac_controller import CapacController
import carla_utils as cu

import os
import cv2
import time
import copy
import threading
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.ion()

import torch
from torch.autograd import grad
import torchvision.transforms as transforms

from learning.model import Generator, EncoderWithV

global_img = None
global_nav = None
global_v0 = 0.
global_vel = 0.
global_plan_time = 0.
global_trajectory = None
start_control = False
global_vehicle = None
global_plan_map = None

global_transform = None
max_steer_angle = 0.
draw_cost_map = None
global_view_img = None
state0 = None

global_collision = False
global_trans_costmap_list = []
global_trans_costmap_dict = {}


MAX_SPEED = 30
img_height = 200
img_width = 400
#longitudinal_length = 25.0 # [m]

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-s', '--save', type=bool, default=False, help='save result')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--vector_dim', type=int, default=64, help='vector dim')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--dt', type=float, default=0.05, help='discretization minimum time interval')
parser.add_argument('--rnn_steps', type=int, default=10, help='rnn readout steps')
args = parser.parse_args()

data_index = args.data
save_path = '/media/wang/DATASET/CARLA/town01/'+str(data_index)+'/'


encoder = EncoderWithV(input_dim=6, out_dim=args.vector_dim).to(device)
# encoder.load_state_dict(torch.load('encoder.pth'))
encoder.load_state_dict(torch.load('/home/cz/Downloads/learning-uncertainty-master/scripts/encoder_e2e.pth'))
encoder.eval()
generator = Generator(input_dim=1+1+args.vector_dim, output=2).to(device)
# generator.load_state_dict(torch.load('generator.pth'))
generator.load_state_dict(torch.load('/home/cz/Downloads/learning-uncertainty-master/scripts/generator_e2e.pth'))
generator.eval()


img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)
        
class Param(object):
    def __init__(self):
        self.longitudinal_length = args.scale
        self.ksize = 21
        
param = Param()
sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)

def mkdir(path):
    os.makedirs(save_path+path, exist_ok=True)

def image_callback(data):
    global state0, global_img, global_plan_time, global_vehicle, global_plan_map,global_nav, global_transform, global_v0
    global_plan_time = time.time()
    global_transform = global_vehicle.get_transform()

    state0 = cu.getActorState('odom', global_plan_time, global_vehicle)
    state0.x = global_transform.location.x
    state0.y = global_transform.location.y
    state0.z = global_transform.location.z
    state0.theta = np.deg2rad(global_transform.rotation.yaw)


    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array
    # try:
    global_nav = get_nav(global_vehicle, global_plan_map)

    v = global_vehicle.get_velocity()
    global_v0 = np.sqrt(v.x**2+v.y**2)
    # except:
        # pass

def view_image_callback(data):
    global global_view_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_view_img = array

def collision_callback(data):
    global global_collision
    global_collision = True

cnt = 0
def visualize(img, nav, curve=None):
    global global_vel, cnt

    text = "speed: "+str(round(3.6*global_vel, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    
    # nav = cv2.resize(nav, (240, 200), interpolation=cv2.INTER_CUBIC)
    # print(img.shape())
    cv2.imshow('Visualization', img)
    cv2.imshow('Nav', nav)
    # if args.save: cv2.imwrite('result/images/img02/'+str(cnt)+'.png', show_img)
    cv2.waitKey(5)
    cnt += 1

def find_nn_ts(ts_list, t):
    if len(ts_list) == 1: return ts_list[0]
    if t <= ts_list[0]: return ts_list[0]
    if t >= ts_list[-1]: return ts_list[-1]
    for i in range(len(ts_list)-1):
        if ts_list[i] < t and t < ts_list[i+1]:
            return ts_list[i] if t-ts_list[i] < ts_list[i+1]-t else ts_list[i+1]
    print('Error in find_nn_ts')
    return ts_list[-1]

def get_traj(plan_time, global_img, global_nav):
    global global_v0, draw_cost_map, state0, global_vehicle
    # state0 = cu.getActorState('odom', plan_time, global_vehicle)
    # state0.x = global_transform.location.x
    # state0.y = global_transform.location.y
    # state0.z = global_transform.location.z
    # state0.theta = np.deg2rad(global_transform.rotation.yaw)
    
    t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
    t.requires_grad = True
    points_num = len(t)

    v = global_v0 if global_v0 > 4 else 4
    v_0 = torch.FloatTensor([v/args.max_speed]).unsqueeze(1)
    v_0 = v_0.to(device)
    condition = torch.FloatTensor([v/args.max_speed]*points_num).view(-1, 1)
    condition = condition.to(device)

    img = Image.fromarray(cv2.cvtColor(global_img,cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(global_nav,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0).to(device)
    # print("input_img shape:", input_img.shape)

    single_latent = encoder(input_img, v_0)
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(1, points_num, single_latent.shape[-1])
    latent = latent.reshape(1 * points_num, single_latent.shape[-1])
    # print("latent shape:", latent.shape)

    output = generator(condition, latent, t)
    # fake_traj = output_xy.data.cpu().numpy()*args.max_dist
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

    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    ax = ax.data.cpu().numpy()
    ay = ay.data.cpu().numpy()
    a = a.data.cpu().numpy()
    #var = np.sqrt(np.exp(log_var))
    
    #state0 = cu.getActorState('odom', plan_time, global_vehicle)
    #time.sleep(0.1)
    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
    return trajectory

def make_plan():
    global global_img, global_nav, global_pcd, global_plan_time, global_trajectory,start_control
    while True:
        plan_time = global_plan_time
        # 1. get cGAN result

        global_trajectory = get_traj(plan_time, global_img, global_nav)

        if not start_control:
            start_control = True
        # time.sleep(1)

            

def get_transform(transform, org_transform):
    x = transform.location.x
    y = transform.location.y
    yaw = transform.rotation.yaw
    x0 = org_transform.location.x
    y0 = org_transform.location.y
    yaw0 = org_transform.rotation.yaw
    
    dx = x - x0
    dy = y - y0
    dyaw = yaw - yaw0
    return dx, dy, dyaw
    
def get_control(x, y, vx, vy, ax, ay):
    global global_vel, max_steer_angle, global_a
    Kx = 0.3
    Kv = 3.0*1.5
    
    Ky = 5.0e-3
    K_theta = 0.10
    
    control = carla.VehicleControl()
    control.manual_gear_shift = True
    control.gear = 1

    v_r = np.sqrt(vx**2+vy**2)

    yaw = np.arctan2(vy, vx)
    theta_e = yaw
    
    #k = (vx*ay-vy*ax)/(v_r**3)
    w_r = (vx*ay-vy*ax)/(v_r**2)
    theta = np.arctan2(y, x)
    dist = np.sqrt(x**2+y**2)
    y_e = dist*np.sin(yaw-theta)
    x_e = dist*np.cos(yaw-theta)
    v_e = v_r - global_vel
    ####################################
    
    #v = v_r*np.cos(theta_e) + Kx*x_eglobal global_trajectory
    w = w_r + v_r*(Ky*y_e + K_theta*np.sin(theta_e))
    
    steer_angle = np.arctan(w*2.405/global_vel) if abs(global_vel) > 0.001 else 0.
    steer = steer_angle/max_steer_angle
    #####################################
    
    #throttle = Kx*x_e + Kv*v_e+0.7
    #throttle = 0.7 +(Kx*x_e + Kv*v_e)*0.06
    #throttle = Kx*x_e + Kv*v_e+0.5
    throttle = Kx*x_e + Kv*v_e + global_a
    # MAGIC !
    #if throttle > 0 and abs(global_vel) < 0.8 and abs(v_r) < 1.0:
    if throttle > 0 and abs(global_vel) < 0.8 and abs(v_r) < 1.2:
        throttle = -1
    
    control.brake = 0.0
    if throttle > 0:
        control.throttle = np.clip(throttle, 0., 1.)
    else:
        #control.brake = np.clip(-0.05*throttle, 0., 1.)
        control.brake = np.clip(abs(100*throttle), 0., 1.)
    control.steer = np.clip(steer, -1., 1.)
    return control

def show_traj(save=False):
    global global_trajectory
    max_x = 30.
    max_y = 30.
    max_speed = 12.0
    while True:
        trajectory = global_trajectory
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = trajectory['x']
        y = trajectory['y']
        ax1.plot(x, y, label='trajectory', color = 'b', linewidth=3)
        ax1.set_xlabel('Tangential/(m)')
        ax1.set_ylabel('Normal/(m)')
        ax1.set_xlim([0., max_x])
        ax1.set_ylim([-max_y, max_y])
        plt.legend(loc='lower right')
        
        t = max_x*np.arange(0, 1.0, 1./x.shape[0])
        a = trajectory['a']
        vx = trajectory['vx']
        vy = trajectory['vy']
        v = np.sqrt(np.power(vx, 2), np.power(vy, 2))
        angle = np.arctan2(vy, vx)/np.pi*max_speed
        ax2 = ax1.twinx()
        ax2.plot(t, v, label='speed', color = 'r', linewidth=2)
        ax2.plot(t, a, label='acc', color = 'y', linewidth=2)
        ax2.plot(t, angle, label='angle', color = 'g', linewidth=2)
        ax2.set_ylabel('Velocity/(m/s)')
        ax2.set_ylim([-max_speed, max_speed])
        plt.legend(loc='upper right')
        # if not save:
        #     plt.show()
        # else:
        #     image = fig2data(fig)
        #     plt.close('all')
        #     return image
    
    
def main():
    global global_nav, global_vel, start_control, global_plan_map, global_vehicle, global_transform, max_steer_angle, global_a, state0, global_collision
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
    global_vehicle = vehicle
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()
    max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)


    spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)

    # prepare map
    destination = carla.Transform()
    destination.location = world.get_random_location_from_navigation()
    global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

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
        'collision':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':collision_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    
    time.sleep(0.5)
    # start to plan
    plan_thread = threading.Thread(target = make_plan, args=())
    # visualization_thread = threading.Thread(target = show_traj, args=())

    while True:
        if (global_img is not None) and (global_nav is not None):
            plan_thread.start()
            break
        else:
            time.sleep(0.001)
    
    # wait for the first plan result
    while not start_control:
        time.sleep(0.001)
    
    # visualization_thread.start()
    # start to control
    print('Start to control')
    
    ctrller = CapacController(world, vehicle, 30)
    while True:
        # change destination
        if close2dest(vehicle, destination):
            #destination = get_random_destination(spawn_points)
            print('get destination !', time.time())
            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()
            # global_plan_map = replan(agent, destination, copy.deepcopy(origin_map)) 
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

        if global_collision:
            global_collision = False
            start_point = random.choice(spawn_points)
            vehicle.set_transform(start_point)
            global_plan_map, destination = replan(agent, destination, copy.deepcopy(origin_map), spawn_points)

            start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
            end_waypoint = agent._map.get_waypoint(destination.location)

            route_trace = agent._trace_route(start_waypoint, end_waypoint)
            start_point.rotation = route_trace[0][0].transform.rotation
            vehicle.set_transform(start_point)

        v = global_vehicle.get_velocity()
        a = global_vehicle.get_acceleration()
        global_vel = np.sqrt(v.x**2+v.y**2+v.z**2)
        global_a = np.sqrt(a.x**2+a.y**2+a.z**2)
        #steer_angle = global_vehicle.get_control().steer*max_steer_angle
        #w = global_vel*np.tan(steer_angle)/2.405
        
        control_time = time.time()
        dt = control_time - global_trajectory['time']
        index = int((dt/args.max_t)//args.dt) +2
        if index > 0.99/args.dt:
            continue
        
        """
        transform = vehicle.get_transform()
        
        dx, dy, dyaw = get_transform(transform, global_transform)
        dyaw = -dyaw
        
        _x = global_trajectory['x'][index] - dx
        _y = global_trajectory['y'][index] - dy
        x = _x*np.cos(dyaw) + _y*np.sin(dyaw)
        y = _y*np.cos(dyaw) - _x*np.sin(dyaw)
        
        _vx = global_trajectory['vx'][index]
        _vy = global_trajectory['vy'][index]
        vx = _vx*np.cos(dyaw) + _vy*np.sin(dyaw)
        vy = _vy*np.cos(dyaw) - _vx*np.sin(dyaw)
        
        _ax = global_trajectory['ax'][index]
        _ay = global_trajectory['ay'][index]
        ax = _ax*np.cos(dyaw) + _ay*np.sin(dyaw)
        ay = _ay*np.cos(dyaw) - _ax*np.sin(dyaw)
        """
        #control = get_control(x, y, vx, vy, ax, ay)
        control = ctrller.run_step(global_trajectory, index, state0)
        vehicle.apply_control(control)

        # x = global_trajectory['x']
        # y = global_trajectory['y']
        # plt.cla()
        # plt.plot(x, y)
        # plt.xlim(-1, 29)
        # plt.ylim(-15, 15)
        # plt.show()
        """
        dyaw = np.deg2rad(global_transform.rotation.yaw)
        x = global_trajectory['x'][index]*np.cos(dyaw) + global_trajectory['y'][index]*np.sin(dyaw)
        y = global_trajectory['y'][index]*np.cos(dyaw) - global_trajectory['x'][index]*np.sin(dyaw)
        """
        #localtion = carla.Location(x = global_transform.location.x+x, y=global_transform.location.y+y, z=2.0)
        #world.debug.draw_point(localtion, size=0.3, color=carla.Color(255,0,0), life_time=10.0)
        
        #print(global_vel*np.tan(control.steer)/w)
        
        curve = None#show_traj(True)
        visualize(global_view_img, global_nav, curve)
        
        #time.sleep(1/60.)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()