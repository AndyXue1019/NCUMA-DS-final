#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import message_filters
import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu, MagneticField
from tf.transformations import euler_from_quaternion

from ekf import EKF


class AttitudeEstimationNode:
    def __init__(self):
        rospy.init_node('attitude_estimation_node')

        self.ekf = EKF()
        self.last_time = None

        self.pose_pub = rospy.Publisher('/imu/pose/filtered', Vector3, queue_size=10)

        imu_sub = message_filters.Subscriber('/imu/data_raw', Imu)
        mag_sub = message_filters.Subscriber('/imu/mag', MagneticField)
        ts = message_filters.TimeSynchronizer([imu_sub, mag_sub], queue_size=10)
        ts.registerCallback(self.sync_callback)

        rospy.loginfo('EKF Node Started with TimeSynchronizer. Waiting for IMU data...')

    def sync_callback(self, imu_msg: Imu, mag_msg: MagneticField):
        curr_time = imu_msg.header.stamp

        # Skip the first frame
        if self.last_time is None:
            self.last_time = curr_time
            return

        dt = (curr_time - self.last_time).to_sec()
        self.last_time = curr_time

        if dt <= 0:
            rospy.logwarn('Non-positive time difference detected, skipping frame.')
            return
        
        raw_gyro = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
        raw_acc = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])
        raw_mag = np.array([
            mag_msg.magnetic_field.x,
            mag_msg.magnetic_field.y,
            mag_msg.magnetic_field.z
        ])

        # Coordinate Transformation
        # New_X = -Raw_Z
        # New_Y = -Raw_Y
        # New_Z = -Raw_X
        gyro = np.array([ -raw_gyro[2], -raw_gyro[1], -raw_gyro[0] ])
        acc  = np.array([ -raw_acc[2],  -raw_acc[1],  -raw_acc[0]  ])
        mag  = np.array([ -raw_mag[2],  -raw_mag[1],  -raw_mag[0]  ])


        self.ekf.predict(gyro, dt)
        self.ekf.update(acc, np.array([0, 0, 1]), self.ekf.R_acc)

        if np.linalg.norm(mag) > 1e-6:
             self.ekf.update(mag, np.array([1, 0, 0]), self.ekf.R_mag)

        # print('Estimated Quaternion:', self.ekf.x)
        self.publish_pose()

    def publish_pose(self):
        q = self.ekf.x

        # tf.transformations expects [x, y, z, w]
        (roll, pitch, yaw) = euler_from_quaternion([q[1], q[2], q[3], q[0]])

        msg = Vector3()
        msg.x = roll
        msg.y = pitch
        msg.z = yaw

        self.pose_pub.publish(msg)


if __name__ == '__main__':
    try:
        node = AttitudeEstimationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Attitude Estimation Node terminated.')
        pass
