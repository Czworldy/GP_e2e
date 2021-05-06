
import numpy as np


def get_leading_vehicle_unsafe(vehicle, vehicles, reference_waypoints, max_distance):
    """
        Get leading vehicle wrt reference_waypoints or global_path.
        !warning: distances between reference_waypoints cannot exceed any vehicle length.
    
    Args:
        reference_waypoints: list of carla.Waypoint
    
    Returns:
        
    """
    
    current_location = vehicle.get_location()
    vehicle_id = vehicle.id
    vehicle_half_height = vehicle.bounding_box.extent.z
    func = lambda loc: loc.distance(current_location)
    obstacles = [(func(o.get_location()), o) for o in vehicles if o.id != vehicle_id and func(o.get_location()) <= 1.001*max_distance]
    sorted_obstacles = sorted(obstacles, key=lambda x:x[0])

    leading_vehicle, leading_distance = None, 0.0
    for i, waypoint in enumerate(reference_waypoints):
        if i > 0: leading_distance += waypoint.transform.location.distance(reference_waypoints[i-1].transform.location)
        if leading_distance > 1.001*max_distance: break
        location = waypoint.transform.location
        location.z += vehicle_half_height
        for _, obstacle in sorted_obstacles:
            obstacle_transform = obstacle.get_transform()
            if obstacle.bounding_box.contains(location, obstacle_transform):
                leading_vehicle = obstacle
                longitudinal_e, _, _ = error_transform(obstacle_transform, waypoint.transform)
                leading_distance += longitudinal_e
                break
        if leading_vehicle is not None: break
    return leading_vehicle, leading_distance



def error_transform(current_transform, target_transform):
    xr, yr, thetar = target_transform.location.x, target_transform.location.y, np.deg2rad(target_transform.rotation.yaw)
    theta_e = pi2pi(np.deg2rad(current_transform.rotation.yaw) - thetar)

    d = (current_transform.location.x - xr, current_transform.location.y - yr)
    t = (np.cos(thetar), np.sin(thetar))

    longitudinal_e, lateral_e = _cal_long_lat_error(d, t)
    return longitudinal_e, lateral_e, theta_e


def _cal_long_lat_error(d, t):
    '''
        Args:
            d, t: array-like
    '''
    dx, dy = d[0], d[1]
    tx, ty = t[0], t[1]
    longitudinal_e = dx*tx + dy*ty
    lateral_e = dx*ty - dy*tx
    return longitudinal_e, lateral_e


def pi2pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


