#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_diff_tf(transform, org_transform):
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