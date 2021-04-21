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
# from utils.capac_controller import CapacController
import carla_utils as cu
from utils import GlobalDict
from utils.gym_wrapper_waypoint import CARLAEnv

from rl.PPO_continuous import Memory, PPO

# import gc
# import objgraph
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

# from learning.model import Generator, EncoderWithV

# import tracemalloc
global_dict = GlobalDict()
global_dict['collision'] = False
global_dict['view_img'] = None
global_dict['vehicle'] = None
global_dict['v0'] = 0.
global_transform = 0.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--name', type=str, default="test", help='name of the script') #rl-train-e2e-08
args = parser.parse_args()

log_path = '/home/cz/result/log/ppo/'+args.name+'/'
# ckpt_path = '/home/cz/result/saved_models/%s' % args.name
logger = SummaryWriter(log_dir=log_path)

sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)


def view_image_callback(data):
    global global_transform
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_dict['view_img'] = array
    v = global_dict['vehicle'].get_velocity()
    global_dict['v0'] = np.sqrt(v.x**2+v.y**2)
    global_transform = global_dict['vehicle'].get_transform()
def collision_callback(data):
    global_dict['collision'] = True

def main():

    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.load_world('Town07')

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
    # physics_control = vehicle.get_physics_control()

    env = CARLAEnv(world, vehicle, global_dict, args)
    # state = env.reset()

    # max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)
    sensor_dict = {
        # 'camera':{
        #     'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
        #     'callback':image_callback,
        #     },
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

    print('Start to control')

    episode_timesteps = 0
    episode_reward = 0
    max_steps = 1e9
    total_steps = 0
    max_episode_steps = 3000
    episode_num = 0

    time_step = 0

    ############## Hyperparameters ##############
    update_timestep = 6000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    is_test = True              # set is test model or not
    #############################################
    state_dim = 30
    action_dim = 1
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    try:
        ppo.policy.load_state_dict(torch.load('/home/cz/result/saved_models/ppo/ppo_waypoint_04/5574_policy.pth'))
        ppo.policy_old.load_state_dict(torch.load('/home/cz/result/saved_models/ppo/ppo_waypoint_04/5574_policy.pth'))
        print('load success')
    except:
        raise ValueError('load model faid')
    while total_steps < max_steps:
        global global_transform
        print("total_episode:", episode_num)
        episode_num += 1
        total_steps += 1
        episode_timesteps = 0
        episode_reward = 0
        total_driving_metre = 0

        state = env.reset()

        for _ in range(max_episode_steps):
            time_step += 1

            action = ppo.select_action(state, memory, is_test=is_test)
            # print(action)


            x_last = global_transform.location.x
            y_last = global_transform.location.y

            next_state, reward, done = env.step(action)


            x_now = global_transform.location.x
            y_now = global_transform.location.y
            driving_metre_in_step = np.sqrt( (x_now - x_last) ** 2 + (y_now - y_last) ** 2)
            total_driving_metre += driving_metre_in_step 
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            # model.replay_buffer.push(state, action, reward, next_state, done)
            if time_step % 200 == 0:
                print("time_step:", time_step)
            if time_step % update_timestep == 0 and is_test == False:
                print("Update policy!")
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
                directory = '/home/cz/result/saved_models/ppo/%s' % (args.name)
                filename = '/home/cz/result/saved_models/ppo/%s/%s_policy.pth' %(args.name, str(episode_num))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(ppo.policy.state_dict(), filename )
            
            # if len(model.replay_buffer) > max(learning_starts, model.batch_size):
            #     print("Start Train")
            #     train_flag = True
            #     # time_s = time.time()
            #     model.train_step(total_steps, noise_std = 0.2, noise_clip = 0.5) #noise_std = 0.2 noise_clip = 0.5
            #     # time_e = time.time()
            #     # print('time:', time_e - time_s)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_timesteps += 1
            if done or episode_timesteps == max_episode_steps:
                # quit()
                logger.add_scalar('episode_reward', episode_reward, episode_num)
                logger.add_scalar('total_driving_metre', total_driving_metre, episode_num)
                #if len(model.replay_buffer) > max(learning_starts, model.batch_size):
                #    for i in range(episode_timesteps):
                #        model.train_step(total_steps, noise_std = 0.1, noise_clip=0.25)
                
                # print(len(model.replay_buffer))
                # if episode_reward > 50:
                #     print('Success')
                # else:
                #     print('Fail')
                # last_episode_reward = episode_reward


                # if episode_num % 20 == 0 and train_flag == True:
                #     directory = '/home/cz/result/saved_models/ppo/%s' % (args.name)
                #     filename = '/home/cz/result/saved_models/ppo/%s/%s_policy.pth' %(args.name, str(episode_num))
                #     if not os.path.exists(directory):
                #         os.makedirs(directory)
                #     torch.save(ppo.policy.state_dict(), filename )


                    # model.save(directory=ckpt_path, filename=str(episode_num)) 
                break
    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()

if __name__ == '__main__':
    main()


