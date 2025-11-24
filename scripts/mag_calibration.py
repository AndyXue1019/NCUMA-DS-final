#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import MagneticField
import numpy as np

class MagCalibrator:
    def __init__(self):
        rospy.init_node('mag_calibrator')
        self.T = 0

        # 初始化 Min/Max 為正負無窮大
        self.mag_min = np.array([float('inf'), float('inf'), float('inf')])
        self.mag_max = np.array([float('-inf'), float('-inf'), float('-inf')])

        # 訂閱磁力計數據 (請確認您的 Topic 名稱)
        self.sub = rospy.Subscriber('/imu/mag', MagneticField, self.callback)

        print("開始校正！請拿著指環在空中畫 '8' 字，並盡量各個角度都轉到...")

    def callback(self, msg):
        self.T += 1
        # 讀取當前磁力
        current_mag = np.array([msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z])

        # 更新最小值與最大值
        self.mag_min = np.minimum(self.mag_min, current_mag)
        self.mag_max = np.maximum(self.mag_max, current_mag)

        # 計算 Hard Iron Bias (偏移量)
        # Bias = (Max + Min) / 2
        hard_iron_bias = (self.mag_max + self.mag_min) / 2.0

        # 計算 Soft Iron Scale (縮放比例)
        # 這是為了讓三個軸的範圍 (Max - Min) 大小一致
        # 1. 計算三個軸各自的範圍
        delta = self.mag_max - self.mag_min
        # 2. 計算平均範圍
        avg_delta = np.sum(delta) / 3.0
        # 3. 計算縮放係數: Scale = 平均範圍 / 該軸範圍
        soft_iron_scale = avg_delta / delta

        # 顯示結果
        print("\n--- Calibration Result ---")
        print(f"T = {self.T}")
        print(f"Bias (Offset): {hard_iron_bias}")
        print(f"Scale        : {soft_iron_scale}")
        print("--------------------------")
        print("請持續轉動，直到數值不再劇烈變化...")

if __name__ == '__main__':
    MagCalibrator()
    rospy.spin()
