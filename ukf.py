import numpy as np
from scipy import linalg
from quaternion import Quaternion  # Assuming the provided quaternion.py file
from Q2 import utils, accel
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import gyroscope
import accel


def initialize_ukf():
    """
    Initialize UKF for orientation tracking with minimal assumptions.

    Returns:
        initial_state: Tuple of (quaternion, angular_velocity)
        initial_covariance: 6x6 state covariance matrix
        process_noise: 6x6 process noise covariance matrix
        measurement_noise: 6x6 measurement noise covariance matrix
    """
    initial_quaternion = Quaternion()  # Identity quaternion
    initial_angular_velocity = np.zeros(3)
    initial_state = (initial_quaternion, initial_angular_velocity)

    orientation_var = 1
    angular_velocity_var = .1

    initial_covariance = np.diag([
        orientation_var, orientation_var, orientation_var,
        angular_velocity_var, angular_velocity_var, angular_velocity_var
    ])

    orientation_process_noise = 1
    angular_velocity_process_noise = .1

    process_noise = np.diag([
        orientation_process_noise, orientation_process_noise,
        orientation_process_noise,
        angular_velocity_process_noise, angular_velocity_process_noise,
        angular_velocity_process_noise
    ])

    accel_noise = 5
    gyro_noise = 5

    measurement_noise = np.diag([
        accel_noise, accel_noise, accel_noise,
        gyro_noise, gyro_noise, gyro_noise
    ])

    return initial_state, initial_covariance, process_noise, measurement_noise


def measurement_model(state):
    """
    Predict accelerometer and gyroscope readings from the state.
    """
    q, omega = state
    g = np.array([0, 0, 9.81])
    g_quat = Quaternion(0, g)
    g_prime_quat = q.inv() * g_quat * q
    g_prime = g_prime_quat.vec()
    accel_prediction = g_prime
    gyro_prediction = omega
    predicted_measurement = np.concatenate([accel_prediction, gyro_prediction])
    return predicted_measurement


def ukf_update_step(prior_mean, prior_covariance, measurement, measurement_noise):
    """
    UKF measurement update.
    """
    sigma_points = generate_sigma_points(prior_mean, prior_covariance)
    measurement_sigma_points = [measurement_model(point) for point in sigma_points]
    measurement_sigma_points = np.array(measurement_sigma_points)
    y_mean = np.mean(measurement_sigma_points, axis=0)
    n = len(sigma_points)
    Y_diff = measurement_sigma_points - y_mean
    P_yy = np.zeros((6, 6))
    for i in range(n):
        diff = Y_diff[i].reshape(-1, 1)
        P_yy += np.dot(diff, diff.T)
    P_yy = P_yy / n + measurement_noise
    P_xy = np.zeros((6, 6))
    for i in range(n):
        q, omega = sigma_points[i]
        q_mean, omega_mean = prior_mean
        q_error = q_mean.inv() * q
        q_error_vector = q_error.axis_angle()
        omega_error = omega - omega_mean
        x_diff = np.concatenate([q_error_vector, omega_error])
        P_xy += np.outer(x_diff, Y_diff[i])
    P_xy = P_xy / n
    K = np.dot(P_xy, np.linalg.inv(P_yy))
    innovation = measurement - y_mean
    q_mean, omega_mean = prior_mean
    correction = np.dot(K, innovation)
    q_correction = correction[:3]
    omega_correction = correction[3:]
    delta_q = Quaternion()
    delta_q.from_axis_angle(q_correction)
    q_updated = q_mean * delta_q
    q_updated.normalize()
    omega_updated = omega_mean + omega_correction
    updated_mean = (q_updated, omega_updated)
    updated_covariance = prior_covariance - np.dot(K, np.dot(P_yy, K.T))
    return updated_mean, updated_covariance


def generate_sigma_points(mean, covariance):
    """
    Generate 2n sigma points from the state mean and 6x6 covariance.
    The first 3 elements of each perturbation vector (from sqrtm) are converted
    to a delta quaternion (axis-angle) and multiplied with the mean quaternion.
    The remaining 3 elements are added to the mean angular velocity.
    """
    q_mean = mean[0]
    omega_mean = mean[1]
    n = covariance.shape[0]
    scaling = np.sqrt(n)
    S = linalg.sqrtm(covariance) * scaling
    sigma_points = []
    for i in range(n):
        S_i = S[:, i]
        delta_orientation = S_i[:3]
        delta_omega = S_i[3:]
        q_delta_pos = Quaternion()
        q_delta_pos.from_axis_angle(delta_orientation)
        q_delta_neg = Quaternion()
        q_delta_neg.from_axis_angle(-delta_orientation)
        q_pos = q_mean * q_delta_pos
        omega_pos = omega_mean + delta_omega
        sigma_points.append((q_pos, omega_pos))
        q_neg = q_mean * q_delta_neg
        omega_neg = omega_mean - delta_omega
        sigma_points.append((q_neg, omega_neg))
    return sigma_points


def sigma_points_gen_UT(x_k_k, P_k_k, Q, dt):
    n = 6
    Sigma_X = np.zeros((7, 2*n))
    Sigma_Y = np.zeros((7, 2*n))
    S = np.linalg.cholesky(P_k_k + Q*dt)
    W_i = np.hstack((np.sqrt(n) * S, -np.sqrt(n) * S))
    q_x = Quaternion(x_k_k[0,0], x_k_k[1:4,0])
    q_x.normalize()
    for j in range(2*n):
        q_W = Quaternion()
        q_W.from_axis_angle(W_i[0:3,j])
        q_X = q_x * q_W
        q_X.normalize()
        Sigma_X[0:4,j] = q_X.q
        Sigma_X[4:7,j] = x_k_k[4:7,0] + W_i[3:6,j]
        q_d = Quaternion()
        angle_d = Sigma_X[4:7,j] * dt
        q_d.from_axis_angle(angle_d)
        q_Y = q_X * q_d
        q_Y.normalize()
        Sigma_Y[0:4,j] = q_Y.q
        Sigma_Y[4:7,j] = Sigma_X[4:7,j]
    return Sigma_X, Sigma_Y


def ukf_prediction_step(prev_mean, mean, covariance, dt, process_noise):
    scaled_process_noise = process_noise * dt
    augmented_covariance = covariance + scaled_process_noise
    sigma_points = generate_sigma_points(mean, augmented_covariance)
    propagated_sigma_points = []
    for sigma_point in sigma_points:
        propagated_point = propagate_dynamics(sigma_point, dt)
        propagated_sigma_points.append(propagated_point)
    q_list = [point[0] for point in propagated_sigma_points]
    omega_list = [point[1] for point in propagated_sigma_points]
    omega_array = np.array(omega_list)
    q_mean, orientation_covariance = quaternion_average(q_list, prev_mean)
    omega_mean = np.mean(omega_array, axis=0)
    predicted_covariance = np.zeros((6, 6))
    predicted_covariance[:3, :3] = orientation_covariance
    n = len(sigma_points)
    for omega in omega_list:
        diff = (omega - omega_mean).reshape(-1, 1)
        predicted_covariance[3:6, 3:6] += np.dot(diff, diff.T) / n
    for i in range(n):
        q_error = q_list[i] * q_mean.inv()
        orientation_error = q_error.axis_angle().reshape(-1, 1)
        omega_error = (omega_list[i] - omega_mean).reshape(-1, 1)
        predicted_covariance[:3, 3:6] += np.dot(orientation_error, omega_error.T) / n
    predicted_covariance[3:6, :3] = predicted_covariance[:3, 3:6].T
    predicted_mean = (q_mean, omega_mean)
    return predicted_mean, predicted_covariance


def propagate_dynamics(state, dt):
    q, omega = state
    omega_mag = np.linalg.norm(omega)
    if omega_mag < 1e-10:
        q_delta = Quaternion()
    else:
        alpha = omega_mag * dt
        axis = omega / omega_mag
        q_delta = Quaternion()
        q_delta.from_axis_angle(axis * alpha)
    q_new = q * q_delta
    q_new.normalize()
    omega_new = omega
    return (q_new, omega_new)


def quaternion_average(quaternions, initial_q=None, max_iterations=100, tolerance=1e-8):
    n = len(quaternions)
    q_mean = quaternions[0]
    E = np.zeros((3, n))
    for iteration in range(max_iterations):
        for i in range(n):
            q_error = quaternions[i] * q_mean.inv()
            q_error.normalize()
            E[:, i] = q_error.axis_angle()
        e_bar = np.mean(E, axis=1)
        if np.linalg.norm(e_bar) < tolerance:
            break
        delta_q = Quaternion()
        delta_q.from_axis_angle(e_bar)
        q_mean = delta_q * q_mean
        q_mean.normalize()
    orientation_covariance = np.zeros((3, 3))
    for i in range(n):
        e_i = E[:, i].reshape(-1, 1)
        orientation_covariance += np.dot(e_i, e_i.T)
    orientation_covariance /= n
    return q_mean, orientation_covariance


if __name__ == "__main__":
    new_accel, gyro, imu_ts, vicon_rot, vicon_ts = utils.load_data(
        data_folder='hw2_p2_data', data_num=2)

    accel_bias = np.array([512.16432647, 500.57738536, 502.43180196])
    accel_sensitivity = np.array([36.25710795, 34.8814697,  34.27137708])
    gyro_bias = np.array([368.05326399, 372.06745886, 375.34814997])
    gyro_sensitivity = np.array([180, 220, 220])

    accel_calibrated = accel.calibrate_accel(new_accel, accel_bias, accel_sensitivity)
    gyro_calibrated = gyroscope.calibrate_gyro(gyro, gyro_bias, gyro_sensitivity)

    state, cov, process_noise, measurement_noise = initialize_ukf()

    num_steps = len(imu_ts)
    quaternions = []
    angular_velocities = []
    covariances = []

    current_state = state
    current_cov = cov
    prev_state = None

    quaternions.append(current_state[0])
    angular_velocities.append(current_state[1])
    covariances.append(current_cov)

    for i in range(1, num_steps, 3):
        if i + 3 < num_steps:
            dt = imu_ts[i + 3] - imu_ts[i - 1]
            accel_measurement = accel_calibrated[:, i + 3]
            gyro_measurement = gyro_calibrated[:, i + 3]
        else:
            dt = imu_ts[-1] - imu_ts[i - 1]
            accel_measurement = accel_calibrated[:, -1]
            gyro_measurement = gyro_calibrated[:, -1]
        measurement = np.concatenate([accel_measurement, gyro_measurement])
        predicted_state, predicted_cov = ukf_prediction_step(prev_state, current_state, current_cov, dt, process_noise)
        current_state, current_cov = ukf_update_step(predicted_state, predicted_cov, measurement, measurement_noise)
        prev_state = current_state
        quaternions.append(current_state[0])
        angular_velocities.append(current_state[1])
        covariances.append(current_cov)

    euler_angles = np.array([q.euler_angles() for q in quaternions])
    vicon_euler_angles = []
    for i in range(vicon_rot.shape[2]):
        q_vicon = Quaternion()
        q_vicon.from_rotm(vicon_rot[:, :, i])
        vicon_euler_angles.append(q_vicon.euler_angles())
    vicon_euler_angles = np.array(vicon_euler_angles)
    ukf_timestamps = []
    for i in range(0, num_steps, 3):
        ukf_timestamps.append(imu_ts[i])
    ukf_timestamps = ukf_timestamps[:len(quaternions)]
    euler_angles = euler_angles[:len(ukf_timestamps)]
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plt.plot(ukf_timestamps, euler_angles[:, 0], 'b-', label='UKF')
    plt.plot(vicon_ts.flatten(), vicon_euler_angles[:, 0], 'r--', label='Vicon')
    plt.title('Roll')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(ukf_timestamps, euler_angles[:, 1], 'b-', label='UKF')
    plt.plot(vicon_ts.flatten(), vicon_euler_angles[:, 1], 'r--', label='Vicon')
    plt.title('Pitch')
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(ukf_timestamps, euler_angles[:, 2], 'b-', label='UKF')
    plt.plot(vicon_ts.flatten(), vicon_euler_angles[:, 2], 'r--', label='Vicon')
    plt.title('Yaw')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ukf_results.png')
    plt.show()