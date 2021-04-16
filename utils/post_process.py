#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def xy2uv(x, y,args):
    pixs_per_meter = args.height/args.scale
    u = (args.height-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+args.width//2).astype(int)
    return u, v

def draw_traj(cost_map, output, args):
    cost_map = Image.fromarray(cost_map).convert("RGB")
    draw = ImageDraw.Draw(cost_map)
    result = output.data.cpu().numpy()
    x = args.max_dist*result[:,0]
    y = args.max_dist*result[:,1]
    u, v = xy2uv(x, y, args)
    for i in range(len(u)-1):
        draw.line((v[i], u[i], v[i+1], u[i+1]), 'red')
        draw.line((v[i]+1, u[i], v[i+1]+1, u[i+1]), 'red')
        draw.line((v[i]-1, u[i], v[i+1]-1, u[i+1]), 'red')
    return cost_map

def add_alpha_channel(img): 
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :int(b_channel.shape[0] / 2)] = 100
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def visualize(global_dict, costmap, args, curve=None):
    img = global_dict['view_img']
    nav = global_dict['nav']
    vel = global_dict['v0']
    #costmap = cv2.cvtColor(costmap,cv2.COLOR_GRAY2RGB)
    text = "speed: "+str(round(3.6*vel, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    
    nav = cv2.resize(nav, (240, 200), interpolation=cv2.INTER_CUBIC)
    down_img = np.hstack([costmap, nav])
    down_img = add_alpha_channel(down_img)
    # print("img shape:", img.shape)
    # print("down_img shape:", down_img.shape)
    # show_img = np.vstack([img, down_img])
    show_img = np.hstack([img, down_img])
    #print(show_img.shape, curve.shape)
    if curve is not None:
        curve = cv2.cvtColor(curve,cv2.COLOR_BGRA2RGBA)
        left_img = cv2.resize(curve, (int(curve.shape[1]*show_img.shape[0]/curve.shape[0]), show_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        show_img = np.hstack([show_img, left_img])
    cv2.imshow('Visualization', show_img)
    if args.save: cv2.imwrite('result/images/img02/'+str(global_dict['cnt'])+'.png', show_img)
    cv2.waitKey(5)
    global_dict['cnt'] += 1
    
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
        if not save:
            plt.show()
        else:
            image = fig2data(fig)
            plt.close('all')
            return image