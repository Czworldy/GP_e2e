 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import random
import copy
import gym
import cv2
import os
from PIL import Image
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
from simulator import config, set_weather, add_vehicle
from agents.navigation.basic_agent import BasicAgent
from .pid import LongPID


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_height = 125
img_width = 400
img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)

nav_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]
nav_trans = transforms.Compose(nav_transforms)

def get_junctions(waypoint_pairs):
    waypoint_ids = []
    waypoints = []
    for wp_pair in waypoint_pairs:
        for wp in wp_pair:
            if wp.id not in waypoint_ids:
                waypoint_ids.append(wp.id)
                waypoints.append(wp)
    return waypoints

def visualize(img, speed, imput, nav):
    text = "speed: "+str(round(3.6*speed, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)    
    cv2.imshow('Visualization', img)
    cv2.imshow('Nav', nav)
    cv2.imshow('input', imput)
    cv2.waitKey(1)
def add_vehicle_tm(client, world, args, traffic_manager):
    vehicles_id_list = []

    blueprints_vehicle = world.get_blueprint_library().filter("vehicle.*")
    # sort the vehicle list by id
    blueprints_vehicle = sorted(blueprints_vehicle, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if args.number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif args.number_of_vehicles >= number_of_spawn_points:
        # msg = 'requested %d vehicles, but could only find %d spawn points'
        # logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
        args.number_of_vehicles = number_of_spawn_points - 1

    # Use command to apply actions on batch of data
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    # this is equal to int 0
    FutureActor = carla.command.FutureActor

    batch = []

    for n, transform in enumerate(spawn_points):
        if n >= args.number_of_vehicles:
            break

        while True:
            blueprint = random.choice(blueprints_vehicle)
            if blueprint.get_attribute('number_of_wheels').as_int() == 4:
                break

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        # set autopilot
        blueprint.set_attribute('role_name', 'autopilot')

        # spawn the cars and set their autopilot all together
        batch.append(SpawnActor(blueprint, transform)
                        .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    # excute the command
    for (i, response) in enumerate(client.apply_batch_sync(batch, True)):
        if response.error:
            # logging.error(response.error)
            # raise ValueError('something wrong')
            print(response.error)
        else:
            # print("Fucture Actor", response.actor_id)
            vehicles_id_list.append(response.actor_id)
    return vehicles_id_list

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

    def __init__(self, client, world, vehicle, global_dict, args):
        self.client = client
        self.world = world
        self.vehicle = vehicle
        self.stop_vehicle = None
        self.agent = BasicAgent(self.vehicle, target_speed=40)
        self.global_dict = global_dict
        self.args = args
        self.pid = LongPID(0.05, -0.5, 1.0)
        self.debugger = self.world.debug

        self.traffic_manager = self.client.get_trafficmanager(8000)
        # every vehicle keeps a distance of 6.0 meter
        self.traffic_manager.set_global_distance_to_leading_vehicle(6.0)
        # Set physical mode only for cars around ego vehicle to save computation
        self.traffic_manager.set_hybrid_physics_mode(True)
        # default speed is 30
        self.traffic_manager.global_percentage_speed_difference(80)  # 80% of 30 km/h
        self.traffic_manager.set_synchronous_mode(True)
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
        self.vehicles_id_list = []
        # environment feedback infomation
        self.state = []
        # self.waypoint = []
        self.done = False
        self.reward = 0.0
        self.input_point = 12
        
        self.seed()
        self.reset()


    def step(self, action, steer, is_test):

        # throttle = action[0].astype("float64")
        how_done = -1
        steer = steer[0].astype("float64") * 0.5
        if abs(steer) < 0.1:
            steer = 0.
        target_kmh = max( (action[0].astype("float64") + 1.) * 10. , 0.)
        target_kmh = min(target_kmh, 20.) 
        if target_kmh < 1.8:
            target_kmh = 0.
        target_ms  = (target_kmh / 3.6) - 0.2

        target_ms *= 1.2

        # print(target_kmh)

        self.reward = 0.
        # current_speed_carla = self.vehicle.get_velocity()
        # current_speed_kmh = np.sqrt(current_speed_carla.x**2+current_speed_carla.y**2) * 3.6
        # throttle, brake = self.pid.run_step(current_speed_kmh, target_kmh)
        for _ in range(2): #4 #50

            if self.global_dict['collision']:
                self.done = True
                self.reward -= 0.1  #-50
                how_done = 0
                print('collision !')
                break
            if close2dest(self.vehicle, self.destination, dist=1.):
                self.done = True
                how_done = 1
                # self.reward += 2.   # reward += 100
                print('Success !')
                break
            current_speed_carla = self.vehicle.get_velocity()
            current_speed_ms = np.sqrt(current_speed_carla.x**2+current_speed_carla.y**2)
            current_speed_kmh = current_speed_ms * 3.6
            throttle, brake = self.pid.run_step(current_speed_ms, target_ms)

            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            self.vehicle.apply_control(control)
            self.world.tick()
            visualize(self.global_dict['view_img'], self.global_dict['v0'], self.global_dict['img'], self.global_dict['nav'])
            # time.sleep(0.01)

        waypoint1, index, diff_rad = self.find_waypoint()
        front_has_car, front_dist = self.find_front_vehicle(index)
        # print(front_dist)

        vehicle_current_x = self.vehicle.get_transform().location.x
        vehicle_current_y = self.vehicle.get_transform().location.y
        vehicle_current_yaw = self.vehicle.get_transform().rotation.yaw

        current_speed_carla = self.vehicle.get_velocity()
        current_speed_kmh = np.sqrt(current_speed_carla.x**2+current_speed_carla.y**2) * 3.6
        # v = self.vehicle.get_velocity()
        # waypoint = self.world_map.get_waypoint(self.vehicle.get_transform().location)

        trace_dist = np.sqrt((waypoint1.location.x - vehicle_current_x) ** 2 + (waypoint1.location.y - vehicle_current_y) ** 2)
        
        
        lane_x = waypoint1.location.x
        lane_y = waypoint1.location.y
        lane_offset = np.sqrt( (lane_x - vehicle_current_x) ** 2 + (lane_y - vehicle_current_y) ** 2) 
        # print("lane_offset: %.2f"  %(lane_offset))
        # if lane_offset > 0.7 and lane_offset <= 1.8:  #转弯的时候还是存在问题 route.py 解决

        # if lane_offset <= 1.2 and np.rad2deg(diff_rad) < :
        #     lane_reward = -0.2*lane_offset+0.1
            
        # # elif lane_offset > 1.5 and lane_offset <= 3.3:
        # #     lane_reward = (0.7 - lane_offset)
        # elif lane_offset > 1.2:
        #     print('off line!')
        #     self.done = True
        #     lane_reward = 0
        #     self.reward -= 2
        # else:
        #     lane_reward = 0
                
        # yaw_reward = -0.2*diff_rad+0.1

        lane_reward = -0.2*lane_offset + 0.1
        yaw_reward  = -0.2*diff_rad + 0.1
        speed_reward = 0.001 * min(current_speed_kmh, 25)
        stop_reward = 0.
        # self.reward += yaw_reward
        # self.reward += lane_reward
        if front_has_car == True:
            speed_reward = - speed_reward * 2.
            if current_speed_kmh < 0.2:
                stop_reward = 0.02
        self.reward += speed_reward
        self.reward += stop_reward

        if lane_offset > 2.0 or np.rad2deg(diff_rad) > 50:
            self.done = True
            how_done = 2
            # self.reward = -2


        print("reward:%.3f , lane_reward:%.2f , trace_dist:%.2f , yaw_reward:%.2f , err_deg:%.2f , speed_reward:%.3f , stop_reword:%.2f" 
                        % (self.reward, lane_reward, trace_dist, yaw_reward, np.rad2deg(diff_rad),speed_reward, stop_reward) )
        # print("trace_dist: %.2f" % (trace_dist))
        # self.state = copy.deepcopy(self.global_dict['img_nav'])
        # vehicle_current_x = self.vehicle.get_transform().location.x
        # vehicle_current_y = self.vehicle.get_transform().location.y
        # vehicle_current_yaw = self.vehicle.get_transform().rotation.yaw

        ################################
        # s = []
        # state_x = []
        # state_y = []
        # state_yaw = []

        # vehicle_current_yaw_rad = self._angle_normalize(np.deg2rad(vehicle_current_yaw))

        # cos_yaw = np.cos(vehicle_current_yaw_rad)
        # sin_yaw = np.sin(vehicle_current_yaw_rad)
        # # print(cos_yaw,sin_yaw,vehicle_current_yaw_rad)

        # s.append(self.route_trace[index][0].transform.location)

        # yaw1 = self._angle_normalize(np.deg2rad(self.route_trace[index][0].transform.rotation.yaw))
        # state_yaw.append(self._angle_normalize(yaw1 - vehicle_current_yaw_rad))

        # if index + 2 < len(self.route_trace) - 1 :
        # for i in range(self.input_point-1):
        #     if index+(1+i)*50 < len(self.route_trace)-1:
        #         # s.append(self.route_trace[index+(i+1)*50][0].transform.location)
        #         # yaw2 = self._angle_normalize(np.deg2rad(self.route_trace[index+(i+1)*50][0].transform.rotation.yaw))
        #         # state_yaw.append(self._angle_normalize(yaw2 - vehicle_current_yaw_rad))
        #         self.debugger.draw_line(self.route_trace[index+50*i][0].transform.location - carla.Location(x=0,y=0,z=0.05),
        #                                  self.route_trace[index+(i+1)*50][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                     life_time=0.6, thickness=1.3)
        #     else:
        #         # s.append(self.route_trace[-1][0].transform.location)
        #         # yaw2 = self._angle_normalize(np.deg2rad(self.route_trace[-1][0].transform.rotation.yaw))
        #         # state_yaw.append(self._angle_normalize(yaw2 - vehicle_current_yaw_rad))
        #         self.debugger.draw_line(self.route_trace[index][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                 self.route_trace[-1][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                 life_time=0.6, thickness=1.3)

            # s.append(self.route_trace[index+2][0].transform.location)
            # yaw3 = self._angle_normalize(np.deg2rad(self.route_trace[index+2][0].transform.rotation.yaw))
            # state_yaw.append(self._angle_normalize(yaw3 - vehicle_current_yaw_rad))
        # else:
        #     s.append(self.route_trace[index][0].transform.location)
        #     state_yaw.append(self._angle_normalize(yaw1 - vehicle_current_yaw_rad))

        #     s.append(self.route_trace[index][0].transform.location)
        #     state_yaw.append(self._angle_normalize(yaw1 - vehicle_current_yaw_rad))
        ################################
        # for s1 in s:
        #     x1 = s1.x - vehicle_current_x
        #     y1 = s1.y - vehicle_current_y
        #     state_x.append((x1 * cos_yaw - y1 * sin_yaw)/10.)
        #     state_y.append((x1 * sin_yaw + y1 * cos_yaw)/10.)
        # for i in range(len(state_yaw)):
        #     state_yaw[i] = state_yaw[i] / np.pi
        
        # self.waypoint = copy.deepcopy(np.array([state_x,state_y,state_yaw]).reshape(30))
        ################################
        # self.state    = copy.deepcopy(self.global_dict['img'])
        img = copy.deepcopy(self.global_dict['img'])
        nav = copy.deepcopy(self.global_dict['nav'])

        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        nav = Image.fromarray(nav)

        img = img_trans(img)
        nav = nav_trans(nav)



        self.state = torch.cat((img, nav), 0)
        if self.done:
            directory = '/home/cz/save_picture/%s' % self.args.name
            if not os.path.exists(directory):
                os.makedirs(directory)
            img = Image.fromarray(cv2.cvtColor(self.global_dict['img'],cv2.COLOR_BGR2RGB))
            img.save('/home/cz/save_picture/%s/%.1f_%d.jpg' % ( self.args.name, time.time(), how_done ),quality=95,subsampling=0)
        # print(self.state)
        # plt.cla()
        # plt.title("waypoint") 
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.plot(self.state[:10],self.state[10:20],'b-+')
        # # plt.show()
        # plt.axis('equal')
        # plt.pause(1)


        return self.state, self.reward, self.done, how_done

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
        if self.vehicles_id_list is not None:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_id_list])
        if self.stop_vehicle is not None:
            self.stop_vehicle.destroy()
        self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        junctions = get_junctions(self.waypoint_tuple_list)

        start_point = random.choice(junctions).transform
        # start_point = self.spawn_points[1]  #1
        # self.vehicle.set_transform(start_point)
        
        for _ in range(2):
            self.world.tick()

        ref_route = get_reference_route(self.world_map, start_point.location, 300, 0.02)
        


        self.destination = ref_route[-1][0].transform
        
        self.global_dict['plan_map'], self.destination, ref_route, start_point= replan(self.world_map, self.vehicle, self.agent, self.destination, copy.deepcopy(self.origin_map), self.spawn_points)
        
        show_plan = cv2.cvtColor(np.asarray(self.global_dict['plan_map']), cv2.COLOR_BGR2RGB)
        cv2.namedWindow('plan_map', 0)    
        cv2.resizeWindow('plan_map', 600, 600)   # 自己设定窗口图片的大小
        cv2.imshow('plan_map', show_plan)
        cv2.waitKey(1)

        self.global_dict['collision'] = False
        
        # start_waypoint = self.agent._map.get_waypoint(self.agent._vehicle.get_location())
        # end_waypoint = self.agent._map.get_waypoint(self.destination.location)

        # self.route_trace = self.agent._trace_route(start_waypoint, end_waypoint)
        self.route_trace = ref_route

        # stop_car_indexes = np.arange(int(len(ref_route)/3), len(ref_route))
        # stop_car_index = np.random.choice(len(stop_car_indexes))
        # # print(stop_car_index)
        # # if self.stop_vehicle is not None:
        # self.stop_vehicle = add_vehicle(self.world, self.world.get_blueprint_library(), vehicle_type='vehicle')
        # self.stop_vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        
        # self.stop_vehicle.set_transform(ref_route[stop_car_index][0].transform)

        # self.vehicles_id_list.append(self.stop_vehicle.id)

        start_point.rotation = self.route_trace[0][0].transform.rotation
        self.vehicle.set_transform(start_point)
        for _ in range(3):
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
            self.world.tick()

        self.vehicles_id_list = add_vehicle_tm(self.client, self.world, self.args, self.traffic_manager)
        ################################

        # vehicle_current_x = self.vehicle.get_transform().location.x
        # vehicle_current_y = self.vehicle.get_transform().location.y
        # vehicle_current_yaw = self.vehicle.get_transform().rotation.yaw

        # index = 0
        # s = []
        # state_x = []
        # state_y = []
        # state_yaw = []

        # vehicle_current_yaw_rad = self._angle_normalize(np.deg2rad(vehicle_current_yaw))

        # cos_yaw = np.cos(vehicle_current_yaw_rad)
        # sin_yaw = np.sin(vehicle_current_yaw_rad)
        # # print(cos_yaw,sin_yaw,vehicle_current_yaw_rad)

        # s.append(self.route_trace[index][0].transform.location)

        # yaw1 = self._angle_normalize(np.deg2rad(self.route_trace[index][0].transform.rotation.yaw))
        # state_yaw.append(self._angle_normalize(yaw1 - vehicle_current_yaw_rad))

        # # if index + 2 < len(self.route_trace) - 1 :
        # for i in range(self.input_point-1):
        #     if index+(1+i)*50 < len(self.route_trace)-1:
        #         # s.append(self.route_trace[index+(i+1)*50][0].transform.location)
        #         # yaw2 = self._angle_normalize(np.deg2rad(self.route_trace[index+(i+1)*50][0].transform.rotation.yaw))
        #         # state_yaw.append(self._angle_normalize(yaw2 - vehicle_current_yaw_rad))
        #         self.debugger.draw_line(self.route_trace[index+50*i][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                 self.route_trace[index+(i+1)*50][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                 life_time=0.6, thickness=1.3)
        #     else:
        #         # s.append(self.route_trace[-1][0].transform.location)
        #         # yaw2 = self._angle_normalize(np.deg2rad(self.route_trace[-1][0].transform.rotation.yaw))
        #         # state_yaw.append(self._angle_normalize(yaw2 - vehicle_current_yaw_rad))
        #         self.debugger.draw_line(self.route_trace[index][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                 self.route_trace[-1][0].transform.location - carla.Location(x=0,y=0,z=0.05), 
        #                                 life_time=0.6, thickness=1.3)

            # s.append(self.route_trace[index+2][0].transform.location)
            # yaw3 = self._angle_normalize(np.deg2rad(self.route_trace[index+2][0].transform.rotation.yaw))
            # state_yaw.append(self._angle_normalize(yaw3 - vehicle_current_yaw_rad))
        # else:
        #     s.append(self.route_trace[index][0].transform.location)
        #     state_yaw.append(self._angle_normalize(yaw1 - vehicle_current_yaw_rad))

        #     s.append(self.route_trace[index][0].transform.location)
        #     state_yaw.append(self._angle_normalize(yaw1 - vehicle_current_yaw_rad))
        ################################

        # for s1 in s:
        #     x1 = s1.x - vehicle_current_x
        #     y1 = s1.y - vehicle_current_y
        #     state_x.append((x1 * cos_yaw - y1 * sin_yaw)/10.)
        #     state_y.append((x1 * sin_yaw + y1 * cos_yaw)/10.)
        # for i in range(len(state_yaw)):
        #     state_yaw[i] = state_yaw[i] / np.pi
        
        # self.waypoint = copy.deepcopy(np.array([state_x,state_y,state_yaw]).reshape(30))
        ################################

        # self.state    = copy.deepcopy(self.global_dict['img'])
        img = copy.deepcopy(self.global_dict['img'])
        nav = copy.deepcopy(self.global_dict['nav'])
        

        if img is not None:
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            nav = Image.fromarray(nav)
            img = img_trans(img)
            nav = nav_trans(nav)
            self.state = torch.cat((img, nav), 0)
        else:
            self.state = None
        self.done = False
        self.reward = 0.0
        return self.state
    
    def find_front_vehicle(self, index):
        position = self.vehicle.get_transform().location
        actor_list = self.world.get_actors()
        front_has_car = False
        dist_list = []
        for actor_id in self.vehicles_id_list:
            current_vehicle = actor_list.find(actor_id)
            if current_vehicle is not None:
                location = current_vehicle.get_transform().location
                x = location.x
                y = location.y
                dist = np.sqrt((x-position.x)**2+(y-position.y)**2)
                if dist < 20.:
                    wp_dist = []
                    for i in range(index, min(index + 400, len(self.route_trace) - 1)):
                        wp_location = self.route_trace[i][0].transform.location
                        wp_dist.append(np.sqrt((x-wp_location.x)**2+(y-wp_location.y)**2))
                    if len(wp_dist):
                        min_val = min(wp_dist)
                        if min_val < 2.:
                            front_has_car = True
                            dist_list.append(dist)
                            print("#########Front has vehicle!##########")
                            # break
        return front_has_car, min(dist_list) if len(dist_list) else -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _angle_normalize(self, angle):
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi
        return angle


        # _, lateral_e, theta_e = agent.global_path.error(agent.get_transform())


        # reward = (-0.2*abs(lateral_e)+0.1) + (-0.2*abs(theta_e)+0.1)

        # if abs(lateral_e) > 1 or abs(theta_e) > np.deg2rad(60):
        #     epoch_info.done = True
        #     reward = -2
        # return reward, epoch_info
