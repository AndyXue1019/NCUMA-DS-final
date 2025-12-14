#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import MagneticField
from visualization_msgs.msg import Marker


class MagVisualizer:
    def __init__(self):
        rospy.init_node('mag_visualizer', anonymous=True)

        # ==========================================
        # [設定區] 請填入您 "目前" 的參數 (用於對比)
        # ==========================================
        self.current_mbias = [10.0, 60.0, 0.0]  # 填入您現在 code 裡的設定
        self.current_magScale = [1.0, 1.0, 1.0]  # 填入您現在 code 裡的設定
        # ==========================================

        # 訂閱磁力計數據
        rospy.Subscriber('/imu/mag', MagneticField, self.mag_callback)

        # 發布 Marker 給 Rviz
        self.marker_pub = rospy.Publisher('/mag_debug/markers', Marker, queue_size=10)

        # 儲存數據點用於即時計算
        self.history_raw = []
        self.max_points = 2000  # 只保留最近 2000 點以免 Rviz 卡頓

        # 用於即時計算 Min/Max
        self.min_vals = np.array([float('inf')] * 3)
        self.max_vals = np.array([float('-inf')] * 3)

        rospy.loginfo(
            'Mag Visualizer Started! Please rotate your IMU in all directions.'
        )

    def mag_callback(self, msg):
        # 1. 取得原始數據 (這裡假設您 topic 發出來的是已經轉好座標軸的，或者是原始的)
        # 建議: 這裡直接讀取原始值，不要經過任何轉換，我們看最原本的樣子
        raw = np.array(
            [msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z]
        )

        # 2. 更新 Min/Max 用於自動計算建議值
        self.min_vals = np.minimum(self.min_vals, raw)
        self.max_vals = np.maximum(self.max_vals, raw)

        # 3. 計算建議的校正參數 (Min-Max 法)
        # Bias = (Max + Min) / 2
        # Scale = Avg_Delta / Delta_Axis
        suggested_bias = (self.max_vals + self.min_vals) / 2.0

        deltas = self.max_vals - self.min_vals
        avg_delta = np.mean(deltas)
        # 避免除以 0
        if np.any(deltas == 0):
            suggested_scale = [1.0, 1.0, 1.0]
        else:
            suggested_scale = avg_delta / deltas

        # 4. 每 50 幀印出一次建議值
        if len(self.history_raw) % 50 == 0:
            print('\n' + '=' * 50)
            print('[即時計算] 轉動越多角度越準確...')
            print(
                f'建議 mbias:    [{suggested_bias[0]:.2f}, {suggested_bias[1]:.2f}, {suggested_bias[2]:.2f}]'
            )
            print(
                f'建議 magScale: [{suggested_scale[0]:.2f}, {suggested_scale[1]:.2f}, {suggested_scale[2]:.2f}]'
            )
            print('=' * 50)

        # 5. 儲存點雲 (為了 Rviz 顯示)
        self.history_raw.append(raw)
        # if len(self.history_raw) > self.max_points:
        #     self.history_raw.pop(0)

        # 6. 發布 Marker
        self.publish_markers()

    def publish_markers(self):
        # --- A. 原始數據點 (紅色) ---
        raw_marker = self.create_marker(0, 1.0, 0.0, 0.0, 'raw_data')

        # --- B. 使用您目前參數校正後的數據點 (綠色) ---
        calib_marker = self.create_marker(1, 0.0, 1.0, 0.0, 'calibrated_data')

        # --- C. 理想球體參考線 (藍色透明) ---
        ref_marker = self.create_sphere_marker(2)

        for p in self.history_raw:
            # 1. 處理原始點 (放大顯示，因為 Tesla 單位很小，通常是微特斯拉 1e-6)
            # 為了在 Rviz 看得清楚，我們統一乘上一個係數，或者直接Normalize
            # 這裡我們選擇: 乘上 100000 (變成以 10uT 為單位) 方便觀察形狀
            VIS_SCALE = 100000.0

            p_raw = Point()
            p_raw.x = p[0] * VIS_SCALE
            p_raw.y = p[1] * VIS_SCALE
            p_raw.z = p[2] * VIS_SCALE
            raw_marker.points.append(p_raw)

            # 2. 處理校正後點
            # 公式: (Raw - Bias) * Scale
            c_x = (p[0] - self.current_mbias[0]) * self.current_magScale[0]
            c_y = (p[1] - self.current_mbias[1]) * self.current_magScale[1]
            c_z = (p[2] - self.current_mbias[2]) * self.current_magScale[2]

            p_cal = Point()
            p_cal.x = c_x * VIS_SCALE
            p_cal.y = c_y * VIS_SCALE
            p_cal.z = c_z * VIS_SCALE
            calib_marker.points.append(p_cal)

        self.marker_pub.publish(raw_marker)
        self.marker_pub.publish(calib_marker)
        self.marker_pub.publish(ref_marker)

    def create_marker(self, id, r, g, b, ns):
        marker = Marker()
        marker.header.frame_id = 'world'  # 確保 Rviz 的 Fixed Frame 是 world
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.2  # 點的大小
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        return marker

    def create_sphere_marker(self, id):
        # 畫一個半徑約為 45uT (典型地磁強度) 的球殼作為參考
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'reference'
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        # 假設地磁約 45 uT -> 0.000045 T
        # 配合上面的 VIS_SCALE = 100000 -> 0.000045 * 100000 = 4.5
        radius = 4.5 * 2
        marker.scale.x = radius
        marker.scale.y = radius
        marker.scale.z = radius
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.3  # 半透明
        marker.pose.orientation.w = 1.0
        return marker


if __name__ == '__main__':
    try:
        MagVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
