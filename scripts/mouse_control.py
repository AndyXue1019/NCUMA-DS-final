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
        self.DEADZONE = 0.04  # Radians
        # Gain: The higher the value, the faster the mouse moves
        self.SPEED_GAIN_X = 40.0
        self.SPEED_GAIN_Y = 40.0

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

        self.is_initialized = False
        self.initialization_time = None
        self.WARMUP_DURATION = 3.0

        self.initial_roll = None
        self.initial_pitch = None

        # x=Roll, y=Pitch, z=Yaw
        rospy.Subscriber('/imu/pose/filtered', Vector3, self.pose_callback)

        rospy.Subscriber('/imu/data_raw', Imu, self.raw_callback)

        rospy.loginfo('Mouse Control Node Started!')

    def pose_callback(self, msg: Vector3):
        """
        Process filtered attitude -> Move mouse
        msg: Vector3 (x=Roll, y=Pitch, z=Yaw) unit: rad
        """
        curr_time = rospy.Time.now()

        if not self.is_initialized:
            self.is_initialized = True
            self.initialization_time = curr_time
            rospy.loginfo('Connection established. Warming up EKF for 3 seconds...')
            return

        elapsed = (curr_time - self.initialization_time).to_sec()

        if elapsed < self.WARMUP_DURATION:
            if int(elapsed * 10) % 10 == 0:
                rospy.loginfo(f'Warming up... {elapsed:.1f}/{self.WARMUP_DURATION}')
            return

        if math.isnan(msg.x) and math.isnan(msg.y):
            rospy.logwarn('Received NaN in attitude data, skipping mouse movement.')
            return

        if self.initial_roll is None:
            self.initial_roll = msg.x
            self.initial_pitch = msg.y
            rospy.loginfo(
                'EKF Settled! Zero point set at Roll: '
                f'{self.initial_roll:.2f}, Pitch: {self.initial_pitch:.2f}'
            )
            rospy.loginfo('Mouse Control ACTIVE!')
            return

        roll = self.normalize_angle(msg.x - self.initial_roll)
        pitch = self.normalize_angle(msg.y - self.initial_pitch)

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

        print(
            f'Roll: {msg.x:.3f} rad, Pitch: {msg.y:.3f} rad\n'
            f'-> Move Mouse: dx={int(-vel_x)}, dy={int(vel_y)}'
        )

        # Move mouse
        try:
            # move(dx, dy) is relative movement
            self.mouse.move(int(-vel_x), int(vel_y))
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

    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi]
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


if __name__ == '__main__':
    try:
        node = MouseControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
