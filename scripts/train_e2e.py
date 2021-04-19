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
from utils.gym_wrapper_e2e import CARLAEnv
from rl.train_e2e_TD3 import TD3

import gc
import objgraph
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

import torch
from torch.autograd import grad
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, EncoderWithV

import tracemalloc
# tracemalloc.start()

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

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--name', type=str, default="rl-train-e2e-09", help='name of the script') #rl-train-e2e-08
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

log_path = '/home/cz/result/log/'+args.name+'/'
ckpt_path = '/home/cz/result/saved_models/%s' % args.name
logger = SummaryWriter(log_dir=log_path)

model = TD3(buffer_size=2e5, noise_decay_steps=3e3, batch_size=32, logger=logger, policy_freq=4)

# model_dict = model.policy_net.state_dict()
# value_dict = model.value_net1.state_dict()

# try:
#     pretrained_dict = torch.load('encoder_e2e.pth')
#     # 1. filter out unnecessary keys
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     pretrained_value_dict = {k: v for k, v in pretrained_dict.items() if k in value_dict}
#     print(pretrained_value_dict.keys())
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict) 
#     value_dict.update(pretrained_value_dict)
#     # 3. load the new state dict
#     model.policy_net.load_state_dict(model_dict)
#     model.value_net1.load_state_dict(value_dict)
#     model.value_net2.load_state_dict(value_dict)
#     print("load model success")
# except:
#     print("load model failed")
#     raise ValueError("load model failed")

# try:
#     # model.policy_net.load_state_dict(torch.load(''))
#     model.policy_net.load_state_dict(torch.load('/home/cz/result/saved_models/rl-train-e2e-07/1220_policy_net.pkl'))
#     model.value_net1.load_state_dict(torch.load('/home/cz/result/saved_models/rl-train-e2e-07/1220_value_net1.pkl'))
#     model.value_net2.load_state_dict(torch.load('/home/cz/result/saved_models/rl-train-e2e-07/1220_value_net2.pkl'))
    
#     print("load success!")
# except:
#     print("load failed!")
#     raise ValueError('load models failed')


# model.policy_net.conv1.requires_grad_(False)
# model.policy_net.conv2.requires_grad_(False)
# model.policy_net.conv3.requires_grad_(False)
# model.policy_net.conv4.requires_grad_(False)
# model.policy_net.bn1.requires_grad_(False)
# model.policy_net.bn2.requires_grad_(False)
# model.policy_net.bn3.requires_grad_(False)

# model.value_net1.conv1.requires_grad_(False)
# model.value_net1.conv2.requires_grad_(False)
# model.value_net1.conv3.requires_grad_(False)
# model.value_net1.conv4.requires_grad_(False)

# model.value_net2.conv1.requires_grad_(False)
# model.value_net2.conv2.requires_grad_(False)
# model.value_net2.conv3.requires_grad_(False)
# model.value_net2.conv4.requires_grad_(False)

img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)

sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)

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
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    blueprint = world.get_blueprint_library()
    # world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')

    global_dict['vehicle'] = vehicle
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()

    env = CARLAEnv(world, vehicle, global_dict, args)
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

    print('Start to control')

    episode_timesteps = 0
    episode_reward = 0
    max_steps = 1e9
    total_steps = 0
    max_episode_steps = 2000
    learning_starts = 3000  #2000
    episode_num = 0

    train_flag = False
    while total_steps < max_steps:
        print("total_episode:", episode_num)
        episode_num += 1
        total_steps += 1
        episode_timesteps = 0
        episode_reward = 0
        total_driving_metre = 0

        state = env.reset()
        # plan_time = time.time()
        # global_dict['state0'] = cu.getActorState('odom', global_plan_time, global_dict['vehicle'])
        # global_dict['state0'].x = global_transform.location.x
        # global_dict['state0'].y = global_transform.location.y
        # global_dict['state0'].z = global_transform.location.z
        # global_dict['state0'].theta = np.deg2rad(global_transform.rotation.yaw)

        # objgraph.show_most_common_types(limit=10)
        # snapshot1 = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')


        for _ in range(max_episode_steps):

            state_cuda = state['img_nav'].unsqueeze(0).to(device)
            action = model.policy_net(state_cuda)
            action = action.detach().cpu().numpy()[0]

            # add_noise = True #if random.random() < 0.9 else False
            # if add_noise:
            action = model.noise.get_action(action)

            x_last = global_transform.location.x
            y_last = global_transform.location.y

            next_state, reward, done, _ = env.step(action)

            # plan_time = ts
            # global_dict['state0'] = cu.getActorState('odom', global_plan_time, global_dict['vehicle'])
            # global_dict['state0'].x = global_transform.location.x
            # global_dict['state0'].y = global_transform.location.y
            # global_dict['state0'].z = global_transform.location.z
            # global_dict['state0'].theta = np.deg2rad(global_transform.rotation.yaw)

            x_now = global_transform.location.x
            y_now = global_transform.location.y
            driving_metre_in_step = np.sqrt( (x_now - x_last) ** 2 + (y_now - y_last) ** 2)
            total_driving_metre += driving_metre_in_step 
            

            model.replay_buffer.push(state, action, reward, next_state, done)
            # print(sys.getsizeof(model.replay_buffer.buffer))
            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
            if len(model.replay_buffer) > max(learning_starts, model.batch_size):
                print("Start Train")
                train_flag = True
                # time_s = time.time()
                model.train_step(total_steps, noise_std = 0.2, noise_clip = 0.5) #noise_std = 0.2 noise_clip = 0.5
                model.train_step(total_steps, noise_std = 0.2, noise_clip = 0.5)
                if len(model.replay_buffer) > 10000:
                    for _ in range(3):
                        model.train_step(total_steps, noise_std = 0.2, noise_clip = 0.5)
                # time_e = time.time()
                # print('time:', time_e - time_s)
            
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
                if episode_num % 20 == 0 and train_flag == True:
                    model.save(directory=ckpt_path, filename=str(episode_num)) 
                break
        # snapshot2 = tracemalloc.take_snapshot()
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        # top = snapshot2.statistics("lineno")

        # print(top_stats[:5])

        # for task in top[:3]:
        #     print(task)
    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()

if __name__ == '__main__':
    main()
