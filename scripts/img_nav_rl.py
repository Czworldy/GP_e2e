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
from learning.models import GeneratorUNet
from learning.path_model import ModelGRU
# from utils import fig2data, add_alpha_channel

from ff_collect_pm_data import sensor_dict
from utils.collect_ipm import InversePerspectiveMapping
from utils.carla_sensor import Sensor, CarlaSensorMaster
from utils.capac_controller import CapacController
import carla_utils as cu
from utils import GlobalDict
from utils.gym_wrapper_img_nav import CARLAEnv
from rl.encoder_RL_TD3 import TD3
import psutil

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
# plt.rcParams.update({'figure.max_open_warning': 0})
# plt.ion()

import torch
from torch.autograd import grad
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, EncoderWithV

global_dict = GlobalDict()
global_dict['img'] = None
global_dict['nav'] = None
global_dict['v0'] = 0.
global_dict['draw_map'] = None
global_dict['vehicle'] = None
global_dict['state0'] = None
global_dict['collision'] = False
global_dict['img_nav'] = None
global_dict['plan_map'] = None
global_dict['view_img'] = None
global_plan_time = 0.
global_trajectory = None
start_control = False

global_transform = None
max_steer_angle = 0.
draw_cost_map = None



MAX_SPEED = 30
img_height = 200
img_width = 400
#longitudinal_length = 25.0 # [m]

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--name', type=str, default="rl-train-img-nav-01", help='name of the script')
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

log_path = '/home/cz/result/log/'+args.name+'/'
ckpt_path = '/home/cz/result/saved_models/%s' % args.name
logger = SummaryWriter(log_dir=log_path)

model = TD3(args=args,buffer_size=3e4, noise_decay_steps=3e3, batch_size=32, logger=logger, policy_freq=4, is_fix_policy_net=True) #48 85
# encoder = EncoderWithV(input_dim=6, out_dim=args.vector_dim).to(device)
model.policy_net.load_state_dict(torch.load('/home/cz/Downloads/learning-uncertainty-master/scripts/encoder.pth'))

generator = Generator(input_dim=1+1+args.vector_dim, output=2).to(device)
generator.load_state_dict(torch.load('/home/cz/Downloads/learning-uncertainty-master/scripts/generator_e2e.pth'))
generator.eval()


img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)
        
# class Param(object):
#     def __init__(self):
#         self.longitudinal_length = args.scale
#         self.ksize = 21
        
# param = Param()
sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)

def mkdir(path):
    os.makedirs(save_path+path, exist_ok=True)

def image_callback(data):
    global global_plan_time, global_transform
    global_plan_time = time.time()
    global_transform = global_dict['vehicle'].get_transform()

    # global_dict['state0'] = cu.getActorState('odom', global_plan_time, global_dict['vehicle'])
    # global_dict['state0'].x = global_transform.location.x
    # global_dict['state0'].y = global_transform.location.y
    # global_dict['state0'].z = global_transform.location.z
    # global_dict['state0'].theta = np.deg2rad(global_transform.rotation.yaw)

    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_dict['img'] = array
    # try:
    global_dict['nav'] = get_nav(global_dict['vehicle'], global_dict['plan_map'])
    img = Image.fromarray(cv2.cvtColor(global_dict['img'],cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(global_dict['nav'],cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    # global_dict['img_nav'] = torch.cat((img, nav), 0).unsqueeze(0).to(device)

    global_dict['img_nav'] = torch.cat((img, nav), 0)
    v = global_dict['vehicle'].get_velocity()
    global_dict['v0'] = np.sqrt(v.x**2+v.y**2)
    global_dict['ts'] = time.time()
    # except:
        # pass

def view_image_callback(data):
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_dict['view_img'] = array

def collision_callback(data):
    global_dict['collision'] = True


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

    t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
    t.requires_grad = True
    points_num = len(t)

    v = global_dict['v0'] if global_dict['v0'] > 4 else 4
    v_0 = torch.FloatTensor([v/args.max_speed]).unsqueeze(1)
    v_0 = v_0.to(device)
    condition = torch.FloatTensor([v/args.max_speed]*points_num).view(-1, 1)
    condition = condition.to(device)

    img = Image.fromarray(cv2.cvtColor(global_dict['img'],cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(global_dict['nav'],cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0).to(device)


    single_latent = encoder(input_img, v_0)
    single_latent = single_latent.unsqueeze(1)
    latent = single_latent.expand(1, points_num, single_latent.shape[-1])
    latent = latent.reshape(1 * points_num, single_latent.shape[-1])

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
    
    #global_dict['state0'] = cu.getActorState('odom', plan_time, global_dict['vehicle'])
    #time.sleep(0.1)
    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
    return trajectory

def make_plan():
    global  global_plan_time, global_trajectory,start_control
    while True:
        plan_time = global_plan_time
        # 1. get cGAN result

        global_trajectory = get_traj(plan_time, global_dict['img'], global_dict['nav'])

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
    global global_transform, max_steer_angle
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.load_world('Town01')

    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,10),
        precipitation=0,
        sun_altitude_angle=random.randint(50,90)
    )
    
    set_weather(world, weather)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)
    
    blueprint = world.get_blueprint_library()
    # world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')

    global_dict['vehicle'] = vehicle
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()

    env = CARLAEnv(world, vehicle, global_dict, args, generator)
    # state = env.reset()

    max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)
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
    
    while True:
        if (global_dict['img'] is not None) and (global_dict['nav'] is not None) and (global_dict['img_nav'] is not None):
            break
        else:
            world.tick()
            # time.sleep(0.01)
    
    # wait for the first plan result
    # while not start_control:
    #     time.sleep(0.001)
    
    print('Start to control')
    
    # ctrller = CapacController(world, vehicle, 30)

    episode_timesteps = 0
    episode_reward = 0
    max_steps = 1e9
    total_steps = 0
    max_episode_steps = 1000
    learning_starts = 2000  #2000
    episode_num = 0

    while total_steps < max_steps:
        print("total_episode:", episode_num)
        episode_num += 1
        total_steps += 1
        episode_timesteps = 0
        episode_reward = 0
        total_driving_metre = 0

        state = env.reset()
        plan_time = time.time()
        global_dict['state0'] = cu.getActorState('odom', global_plan_time, global_dict['vehicle'])
        global_dict['state0'].x = global_transform.location.x
        global_dict['state0'].y = global_transform.location.y
        global_dict['state0'].z = global_transform.location.z
        global_dict['state0'].theta = np.deg2rad(global_transform.rotation.yaw)

        add_noise = False #if random.random() < 0.3 else False

        for step in range(max_episode_steps):
            # t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
            # t.requires_grad = True
            t = np.arange(0, 0.99, args.dt)
            points_num = len(t)

            # v = global_dict['v0'] if global_dict['v0'] > 4 else 4
            # v_0 = torch.FloatTensor([v/args.max_speed]).unsqueeze(1)
            # v_0 = v_0.to(device)

            condition = torch.FloatTensor([state['v0']/args.max_speed]*points_num).view(-1, 1)
            condition = condition.to(device)

            input_img = state['img_nav'].unsqueeze(0).to(device)
            v_0 = torch.FloatTensor([state['v0']/args.max_speed]).unsqueeze(1).to(device)

            action = model.policy_net(input_img, v_0)
            action = action.detach().cpu().numpy()[0]

            if add_noise:
                action = model.noise.get_action(action)

            x_last = global_transform.location.x
            y_last = global_transform.location.y

            next_state, reward, done, ts = env.step(action, plan_time, condition)

            plan_time = ts
            global_dict['state0'] = cu.getActorState('odom', global_plan_time, global_dict['vehicle'])
            global_dict['state0'].x = global_transform.location.x
            global_dict['state0'].y = global_transform.location.y
            global_dict['state0'].z = global_transform.location.z
            global_dict['state0'].theta = np.deg2rad(global_transform.rotation.yaw)

            x_now = global_transform.location.x
            y_now = global_transform.location.y
            driving_metre_in_step = np.sqrt( (x_now - x_last) ** 2 + (y_now - y_last) ** 2)
            total_driving_metre += driving_metre_in_step 

            model.replay_buffer.push(state, action, reward, next_state, done)
            print(sys.getsizeof(model.replay_buffer.buffer))
            print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
            if len(model.replay_buffer) > max(learning_starts, model.batch_size):
                # print("Start Train:")
                time_s = time.time()
                model.train_step(total_steps, noise_std = 0.1, noise_clip = 0.05) #noise_std = 0.2 noise_clip = 0.5
                time_e = time.time()
                print('time:', time_e - time_s)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_timesteps += 1
            if done or episode_timesteps == max_episode_steps:
                logger.add_scalar('episode_reward', episode_reward, episode_num)
                logger.add_scalar('total_driving_metre', total_driving_metre, episode_num)
                #if len(model.replay_buffer) > max(learning_starts, model.batch_size):
                #    for i in range(episode_timesteps):
                #        model.train_step(total_steps, noise_std = 0.1, noise_clip=0.25)
                
                # print(len(model.replay_buffer))
                if episode_reward > 50:
                    print('Success')
                else:
                    print('Fail')
                # last_episode_reward = episode_reward
                if episode_num % 5 == 0:
                    model.save(directory=ckpt_path, filename=str(episode_num)) 
                break
  
    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()