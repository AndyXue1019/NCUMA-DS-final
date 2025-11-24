#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np
import rosbag


def calculate_R(
    bag_file: str, imu_topic: str = '/imu/data_raw', mag_topic: str = '/imu/mag'
):
    print(f'正在處理 Rosbag 檔案: {bag_file} ...')

    acc_data = []  # [x, y, z]
    mag_data = []  # [x, y, z]
    gyro_data = []  # [x, y, z] (雖然主要用於 Q，但也可以參考)

    try:
        bag = rosbag.Bag(bag_file)
    except Exception as e:
        print(f'無法開啟檔案: {e}')
        return

    count = 0
    for topic, msg, _ in bag.read_messages(topics=[imu_topic, mag_topic]):
        if topic == imu_topic:
            acc_data.append(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ]
            )
            gyro_data.append(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            )
        elif topic == mag_topic:
            mag_data.append(
                [msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z]
            )
        count += 1

    bag.close()

    print(f'讀取完成，共 {count} 筆訊息。')
    print(f'Accel 樣本數: {len(acc_data)}')
    print(f'Mag   樣本數: {len(mag_data)}')
    if len(acc_data) == 0 or len(mag_data) == 0:
        print('錯誤: 沒有讀取到足夠的數據，請檢查 Topic 名稱是否正確。')
        return

    np_acc = np.array(acc_data)
    np_mag = np.array(mag_data)
    np_gyro = np.array(gyro_data)

    var_acc = np.var(np_acc, axis=0)
    var_mag = np.var(np_mag, axis=0)
    var_gyro = np.var(np_gyro, axis=0)

    print('\n' + '=' * 50)
    print('  RESULT: Measurement Noise Covariance (R)')
    print('=' * 50)

    print('\n[加速度計 R_acc] (單位: (m/s^2)^2)')
    print(f'X axis variance: {var_acc[0]:.8f}')
    print(f'Y axis variance: {var_acc[1]:.8f}')
    print(f'Z axis variance: {var_acc[2]:.8f}')
    print('-' * 30)
    print('可以直接複製到 Python code:')
    print(f'self.R_acc = np.diag([{var_acc[0]:.6f}, {var_acc[1]:.6f}, {var_acc[2]:.6f}])')

    print('\n' + '-' * 50)

    print('\n[磁力計 R_mag] (單位: T^2)')
    print(f'X axis variance: {var_mag[0]:.12f}')
    print(f'Y axis variance: {var_mag[1]:.12f}')
    print(f'Z axis variance: {var_mag[2]:.12f}')
    print('-' * 30)
    print('可以直接複製到 Python code:')
    print(f'self.R_mag = np.diag([{var_mag[0]:.10f}, {var_mag[1]:.10f}, {var_mag[2]:.10f}])')

    print('\n' + '-' * 50)

    print('\n[參考用: 陀螺儀變異數] (單位: (rad/s)^2)')
    print(f'Gyro variance: {var_gyro}')
    print('注意: 這通常用於設定 Q (Process Noise)，但 Q 通常需要手動調大於此值以容忍模型誤差。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='計算 ROS Bag 中 IMU 數據的 Covariance (R Matrix)'
    )
    parser.add_argument('bagfile', help='輸入的 .bag 檔案路徑')
    args = parser.parse_args()

    calculate_R(args.bagfile)
