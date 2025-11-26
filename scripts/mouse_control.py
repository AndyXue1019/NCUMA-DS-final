#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import numpy as np
import rospy
from geometry_msgs.msg import Vector3

# mouse control library
from pynput.mouse import Button, Controller
from sensor_msgs.msg import Imu


class MouseControlNode:
    def __init__(self):
        rospy.init_node('mouse_control_node', anonymous=True)

        # Deadzone: Prevent small movements from affecting the mouse
        self.DEADZONE = 0.08  # 約 4.5 度
        # Gain: The higher the value, the faster the mouse moves
        self.SPEED_GAIN_X = 1500.0
        self.SPEED_GAIN_Y = 1500.0

        # [Click Parameters]
        # Jerk threshold: Acceleration change above this value is considered a tap
        # Unit is m/s^2 (assuming your raw data is in m/s^2)
        # Recommended value: 8.0 ~ 15.0, depending on the force applied
        self.TAP_THRESHOLD = 10.0
        # Cooldown time: Prevent multiple triggers from a single tap (seconds)
        self.CLICK_COOLDOWN = 0.4
        # Right click trigger angle: When Roll tilts to the right beyond this angle (radians), a tap = right click
        self.RIGHT_CLICK_ROLL_THRESHOLD = 0.5  # About 30 degrees

        self.mouse = Controller()
        self.last_accel = None
        self.last_click_time = rospy.Time.now()
        self.current_roll = 0.0  # For right click detection

        # x=Roll, y=Pitch, z=Yaw
        rospy.Subscriber('/imu/pose/filtered', Vector3, self.pose_callback)

        rospy.Subscriber('/imu/data_raw', Imu, self.raw_callback)

        rospy.loginfo('Mouse Control Node Started!')
        rospy.loginfo('Mode: Joystick Control (Tilt to move)')

    def pose_callback(self, msg: Vector3):
        """
        Process filtered attitude -> Move mouse
        msg: Vector3 (x=Roll, y=Pitch, z=Yaw) unit: rad
        """
        roll = msg.x
        pitch = msg.y

        self.current_roll = roll

        # Deadzone processing
        roll = (
            0.0
            if abs(roll) < self.DEADZONE
            else roll - (math.copysign(self.DEADZONE, roll))
        )
        pitch = (
            0.0
            if abs(pitch) < self.DEADZONE
            else pitch - (math.copysign(self.DEADZONE, pitch))
        )

        # Calculate speed
        vel_x = math.copysign(roll**2, roll) * self.SPEED_GAIN_X
        vel_y = math.copysign(pitch**2, pitch) * self.SPEED_GAIN_Y

        # Move mouse
        try:
            # move(dx, dy) is relative movement
            self.mouse.move(int(vel_x), int(-vel_y))
        except Exception:
            pass

    def raw_callback(self, msg: Imu):
        """
        Process raw IMU data -> Detect taps -> Trigger mouse clicks
        msg: Imu
        """
        curr_accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )

        # Initialize the first frame
        if self.last_accel is None:
            self.last_accel = curr_accel
            return

        # Calculate  jerk (change in acceleration)
        jerk = np.linalg.norm(curr_accel - self.last_accel)

        self.last_accel = curr_accel

        # Check for tap
        now = rospy.Time.now()
        time_diff = (now - self.last_click_time).to_sec()

        if jerk > self.TAP_THRESHOLD and time_diff > self.CLICK_COOLDOWN:
            self.perform_click()
            self.last_click_time = now

    def perform_click(self):
        """
        Perform mouse click (left or right)
        """
        # If Roll exceeds the right click threshold, perform right click
        # Otherwise, perform left click
        if self.current_roll > self.RIGHT_CLICK_ROLL_THRESHOLD:
            rospy.loginfo(f'RIGHT CLICK! (Jerk detected, Roll={self.current_roll:.2f})')
            self.mouse.click(Button.right, 1)
        else:
            rospy.loginfo('LEFT CLICK! (Jerk detected)')
            self.mouse.click(Button.left, 1)

if __name__ == '__main__':
    try:
        node = MouseControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass