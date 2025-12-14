#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class EKF:
    def __init__(self):
        # State vector: [q0, q1, q2, q3] (quaternion)
        self.x = np.array([1.0, 0.0, 0.0, 0.0])
        # Covariance matrix
        self.P = np.eye(4) * 0.1
        # Process noise covariance
        self.Q = np.eye(4) * 0.001
        # Measurement noise covariance
        self.R_acc = np.diag([0.000439, 0.000406, 0.001112])
        self.R_mag = np.diag([1e-2, 1e-2, 1e-2])

    def predict(self, gyro, dt):
        '''
        Predict the next state using gyroscope data.
        gyro: [wx, wy, wz] (rad/s)
        dt: time step (s)
        '''
        wx, wy, wz = gyro

        Omega = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])

        F = np.eye(4) + 0.5 * Omega * dt

        self.x = np.dot(F, self.x)

        self.x /= np.linalg.norm(self.x)

        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, measurement: np.ndarray, reference_vector: np.ndarray, R_noise: np.ndarray):
        '''
        Common update step for both accelerometer and magnetometer.
        measurement: measured vector [mx, my, mz]
        reference_vector: reference vector in the inertial frame
        R_noise: measurement noise covariance matrix
        '''
        # Normalize measurement and reference vector
        meas_norm = np.linalg.norm(measurement)
        if meas_norm < 1e-6:
            print('Invalid measurement vector, skipping update.')
            return  # Invalid measurement, skip update
        z = measurement / meas_norm
        ref = reference_vector / np.linalg.norm(reference_vector)

        # h(x) = R(q).T * ref
        h_x = self.rotate_vector_by_quaternion_inverse(ref, self.x)

        # 2. Jacobian H (partial h(x))
        H = self.calculate_jacobian(ref, self.x)

        # 3. Kalman Gain (K)
        # S = H * P * H.T + R
        S = np.dot(np.dot(H, self.P), H.T) + R_noise
        # K = P * H.T * inv(S)
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # 4. x = x + K * (z - h(x))
        y = z - h_x # Residual
        self.x = self.x + np.dot(K, y)
        
        # Normalize again
        self.x /= np.linalg.norm(self.x)

        # 5. Update covariance P = (I - K * H) * P
        I_ = np.eye(4)
        self.P = np.dot((I_ - np.dot(K, H)), self.P)

    def rotate_vector_by_quaternion_inverse(self, v, q):
        '''
        Rotate vector v by the inverse of quaternion q.
        v: 3D vector
        q: quaternion [qw, qx, qy, qz]
        '''
        qw, qx, qy, qz = q
        vx, vy, vz = v

        rw =  qx*vx + qy*vy + qz*vz
        rx =  qw*vx - qz*vy + qy*vz
        ry =  qw*vy + qz*vx - qx*vz
        rz =  qw*vz - qy*vx + qx*vy

        tx =  rw*qx + rx*qw + ry*qz - rz*qy
        ty =  rw*qy - rx*qz + ry*qw + rz*qx
        tz =  rw*qz + rx*qy - ry*qx + rz*qw
        return np.array([tx, ty, tz])
    
    def calculate_jacobian(self, ref, q):
        '''
        Calculate the Jacobian matrix H (3x4). 
        This is h(x) partial derivative with respect to quaternion q.
        ref: reference vector in inertial frame [rx, ry, rz]
        q: quaternion [qw, qx, qy, qz]
        '''
        rx, ry, rz = ref
        qw, qx, qy, qz = q

        H = np.zeros((3, 4))
        
        # Case 1: 重力 (Accelerometer) - Ref [0, 0, 1]
        if np.allclose(ref, [0, 0, 1]):
             H = 2 * np.array([
                [-qy,  qz, -qw,  qx],
                [ qx,  qw,  qz,  qy],
                [ qw, -qx, -qy,  qz]
            ])
            
        # Case 2: 磁北 (Magnetometer) - Ref [1, 0, 0] 
        elif np.allclose(ref, [1, 0, 0]):
            H = 2 * np.array([
                [ 0,   0,  -2*qy, -2*qz],
                [-qz,  qy,  qx,   -qw  ],
                [ qy,  qz,  qw,    qx  ]
            ])
            
        # Case 3: Fallback (Generic formula)
        # 如果不是上述兩個向量，使用通用公式
        else:
            rx, ry, rz = ref
            H = 2 * np.array([
                [ qx*rx + qy*ry + qz*rz,  qw*rx - qz*ry + qy*rz,  qz*rx + qw*ry - qx*rz, -qy*rx + qx*ry + qw*rz],
                [-qz*rx + qw*ry - qx*rz,  qy*rx - qx*ry - qw*rz,  qw*rx + qz*ry - qy*rz, -qx*rx - qy*ry + qz*rz],
                [ qy*rx - qx*ry - qw*rz,  qz*rx - qw*ry + qx*rz,  qw*rx - qz*ry + qy*rz,  qx*rx + qy*ry + qz*rz]
            ])
            
        return H