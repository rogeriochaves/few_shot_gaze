#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

# import gi.repository
# gi.require_version('Gdk', '3.0')
# from gi.repository import Gdk
# from PyQt5 import QtGui
import pyautogui
import numpy as np

class monitor:

    def __init__(self):
        # display = Gdk.Display.get_default()
        # screen = display.get_default_screen()
        # default_screen = screen.get_default()
        # num = default_screen.get_number()

        self.retina_multiplier = 2

        self.h_mm = 1850 # 185 during training? # default_screen.get_monitor_height_mm(num)
        self.w_mm = 1950 # 195 during training? # default_screen.get_monitor_width_mm(num)

        screenWidth, screenHeight = pyautogui.size()
        self.w_pixels = screenWidth * self.retina_multiplier
        self.h_pixels = screenHeight * self.retina_multiplier

    def monitor_to_camera(self, x_pixel, y_pixel):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
        y_cam_mm = 10.0 + (y_pixel/self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)
        x = max(4, min(x_mon_pixel[0][0], self.w_pixels - 4)) / self.retina_multiplier
        y = max(4, min(y_mon_pixel[0][0] - 90, self.h_pixels - 4)) / self.retina_multiplier
        print("camera_to_monitor", x, y)

        return x, y
