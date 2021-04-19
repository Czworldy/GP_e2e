
import random
import simulator
import sys
simulator.load('/home/czworldy/CARLA_0.9.9.4')
import carla
sys.path.append('/home/czworldy/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.local_planner import RoadOption
def get_reference_route(town_map, location, distance_range, sampling_resolution):
    distance_range, sampling_resolution = float(distance_range), float(sampling_resolution)
    sampling_number = int(distance_range / sampling_resolution) + 1
    waypoint = get_waypoint(town_map, location)
    return get_reference_route_wrt_waypoint(waypoint, sampling_resolution, sampling_number)

def get_reference_route_wrt_waypoint(waypoint, sampling_resolution, sampling_number):
    # random.seed(1)
    next_waypoint = waypoint
    reference_route = [(next_waypoint, RoadOption.LANEFOLLOW)]
    for i in range(1, sampling_number):
        next_waypoint = random.choice(next_waypoint.next(sampling_resolution))
        reference_route.append( (next_waypoint, RoadOption.LANEFOLLOW) )
    return reference_route

def get_waypoint(town_map, location):
    # location = vehicle.get_location()
    return town_map.get_waypoint(location)
