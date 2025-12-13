#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import message_filters
import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu, MagneticField
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from ekf import EKF


class AttitudeEstimationNode:
    def __init__(self):
        rospy.init_node('attitude_estimation_node')

        self.ekf = EKF()
        self.last_time = None

        self.pose_pub = rospy.Publisher('/imu/pose/filtered', Vector3, queue_size=10)
        self.marker_pub = rospy.Publisher('/imu/marker', Marker, queue_size=10)

        imu_sub = message_filters.Subscriber('/imu/data_raw', Imu)
        mag_sub = message_filters.Subscriber('/imu/mag', MagneticField)
        ts = message_filters.TimeSynchronizer([imu_sub, mag_sub], queue_size=10)
        ts.registerCallback(self.sync_callback)

        # Gyro bias calibration
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.is_calibrated = False
        self.calibration_buffer = []
        self.CALIBRATION_SAMPLES = 100  # Number of samples for calibration

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

        raw_gyro = np.array(
            [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ]
        )
        raw_acc = np.array(
            [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
            ]
        )
        raw_mag = np.array(
            [
                mag_msg.magnetic_field.x,
                mag_msg.magnetic_field.y,
                mag_msg.magnetic_field.z,
            ]
        )

        if not self.is_calibrated:
            self.calibration_buffer.append(raw_gyro)
            if len(self.calibration_buffer) >= self.CALIBRATION_SAMPLES:
                # 計算平均值作為 Bias
                self.gyro_bias = np.mean(self.calibration_buffer, axis=0)
                self.is_calibrated = True
                rospy.loginfo(f'Calibration Done! Gyro Bias: {self.gyro_bias}')

                # 印出原始加速度，確認軸向
                rospy.loginfo('========================================')
                rospy.loginfo(f'Raw Accel when static: {raw_acc}')
                rospy.loginfo('========================================')
            else:
                if len(self.calibration_buffer) % 20 == 0:
                    rospy.loginfo(f'Calibrating... {len(self.calibration_buffer)}/{self.CALIBRATION_SAMPLES}')
                return  # 校正期間不執行 EKF

        gyro_corrected = raw_gyro - self.gyro_bias

        # Coordinate Transformation
        gyro = np.array([-gyro_corrected[2], -gyro_corrected[1], -gyro_corrected[0]])
        acc = np.array([-raw_acc[2], -raw_acc[1], -raw_acc[0]])
        mag = np.array([-raw_mag[2], -raw_mag[1], -raw_mag[0]])

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

        self.publish_marker(q)

    def publish_marker(self, q):
        marker = Marker()
        # [重要] Frame ID 設為 "world"，等一下 Rviz 的 Fixed Frame 也要設成這個
        marker.header.frame_id = 'world'
        marker.header.stamp = rospy.Time.now()

        # 設定 Marker 類型為 CUBE (立方體)
        marker.ns = 'imu_shape'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 設定位置 (固定在原點)
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0

        # 設定姿態 (從 EKF 取得的四元數)
        # 注意: EKF 的 q 是 [w, x, y, z]，ROS msg 是 x, y, z, w
        marker.pose.orientation.w = q[0]
        marker.pose.orientation.x = q[1]
        marker.pose.orientation.y = q[2]
        marker.pose.orientation.z = q[3]

        # 設定尺寸 (單位: 公尺)
        # 做成扁平狀，模擬 IMU 模組的外觀
        marker.scale.x = 0.1  # 長 5cm (X軸-紅色)
        marker.scale.y = 0.5  # 寬 3cm (Y軸-綠色)
        marker.scale.z = 0.3  # 高 1cm (Z軸-藍色)

        # 設定顏色 (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0  # 不透明

        self.marker_pub.publish(marker)


if __name__ == '__main__':
    try:
        node = AttitudeEstimationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Attitude Estimation Node terminated.')
        pass
