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
    # Initialize state with identity quaternion (no rotation) and zero angular velocity
    initial_quaternion = Quaternion()  # Default constructor creates identity quaternion
    initial_angular_velocity = np.zeros(3)
    initial_state = (initial_quaternion, initial_angular_velocity)

    # Initialize state covariance (uncertainty in initial state)
    # Using moderate uncertainty values
    orientation_var = .001  # uncertainty in orientation (rad^2)
    angular_velocity_var = .001  # uncertainty in angular velocity (rad/s)^2

    initial_covariance = np.diag([
        orientation_var, orientation_var, orientation_var,
        angular_velocity_var, angular_velocity_var, angular_velocity_var
    ])

    # Process noise covariance (how much uncertainty is added per second)
    # Using conservative values
    orientation_process_noise = 0.01  # rad^2/s
    angular_velocity_process_noise = .1 # (rad/s)^2/s

    process_noise = np.diag([
        orientation_process_noise, orientation_process_noise,
        orientation_process_noise,
        angular_velocity_process_noise, angular_velocity_process_noise,
        angular_velocity_process_noise
    ])

    # Measurement noise covariance (uncertainty in sensor readings)
    accel_noise = 2  # (m/s^2)^2
    gyro_noise = 0.05  # (rad/s)^2

    measurement_noise = np.diag([
        accel_noise, accel_noise, accel_noise,
        gyro_noise, gyro_noise, gyro_noise
    ])

    return initial_state, initial_covariance, process_noise, measurement_noise


def measurement_model(state):
    """
    Apply measurement model to predict accelerometer and gyroscope readings.

    Args:
        state: Tuple of (quaternion, angular_velocity)

    Returns:
        predicted_measurement: Array of [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    """
    q, omega = state

    # Gravity vector in world frame (pointing down)
    g = np.array([0, 0, 9.81])

    # Apply the measurement model: g′ = q^(-1) * g * q
    # First create a quaternion representation of g (with scalar=0)
    g_quat = Quaternion(0, g)

    # Compute g' = q^(-1) * g * q
    # !!!THIS COULD BE WRONG NOT USING THE B' EQUATION
    g_prime_quat = q.inv() * g_quat * q
    g_prime = g_prime_quat.vec()

    # The accelerometer measures -g' in the absence of linear acceleration
    accel_prediction = -g_prime

    # Gyroscope directly measures angular velocity
    gyro_prediction = omega

    # Combine predictions into measurement vector
    predicted_measurement = np.concatenate([accel_prediction, gyro_prediction])

    return predicted_measurement


def ukf_update_step(prior_mean, prior_covariance, measurement,
                    measurement_noise):
    """
    Perform UKF update step (measurement update).

    Args:
        prior_mean: Prior state mean (quaternion, angular_velocity)
        prior_covariance: Prior state covariance (6x6 matrix)
        measurement: Observed measurement [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        measurement_noise: Measurement noise covariance (6x6 matrix)

    Returns:
        updated_mean: Updated state mean
        updated_covariance: Updated state covariance
    """
    # Generate sigma points from prior distribution
    sigma_points = generate_sigma_points(prior_mean, prior_covariance)

    # Transform sigma points through measurement model
    measurement_sigma_points = [measurement_model(point) for point in
                                sigma_points]
    measurement_sigma_points = np.array(measurement_sigma_points)

    # Compute predicted measurement mean
    y_mean = np.mean(measurement_sigma_points, axis=0)

    # Compute measurement covariance
    n = len(sigma_points)
    Y_diff = measurement_sigma_points - y_mean

    # Calculate measurement covariance
    P_yy = np.zeros((6, 6))
    for i in range(n):
        diff = Y_diff[i].reshape(-1, 1)
        P_yy += np.dot(diff, diff.T)

    # SHOULD WE BE DIVIDING BY N HERE
    P_yy = P_yy / n + measurement_noise

    # Compute cross-covariance between state and measurement
    P_xy = np.zeros((6, 6))

    for i in range(n):
        # Compute state error vector
        q, omega = sigma_points[i]
        q_mean, omega_mean = prior_mean

        # Compute error quaternion (relative rotation)
        q_error = q_mean.inv() * q

        q_error_vector = q_error.axis_angle()

        # Compute error in angular velocity
        omega_error = omega - omega_mean

        # Combine errors
        x_diff = np.concatenate([q_error_vector, omega_error])

        # Compute cross-covariance term
        P_xy += np.outer(x_diff, Y_diff[i])

    P_xy = P_xy / n

    # Compute Kalman gain
    K = np.dot(P_xy, np.linalg.inv(P_yy))

    # Calculate innovation
    innovation = measurement - y_mean

    # Update mean
    q_mean, omega_mean = prior_mean

    # Extract correction components
    correction = np.dot(K, innovation)
    q_correction = correction[:3]
    omega_correction = correction[3:]

    # Apply quaternion correction
    delta_q = Quaternion()
    delta_q.from_axis_angle(q_correction)
    q_updated = q_mean * delta_q
    q_updated.normalize()

    # Apply angular velocity correction
    omega_updated = omega_mean + omega_correction

    updated_mean = (q_updated, omega_updated)

    # Update covariance
    updated_covariance = prior_covariance - np.dot(K, np.dot(P_yy, K.T))

    return updated_mean, updated_covariance


def generate_sigma_points(mean, covariance, kappa=0):
    """
    Generate sigma points from state mean and covariance.

    Args:
        mean: Tuple of (quaternion, angular_velocity) where:
            - quaternion is a Quaternion object
            - angular_velocity is a numpy array of shape (3,)
        covariance: State covariance matrix of shape (6,6)
        kappa: Scaling parameter (default=0)

    Returns:
        List of sigma points, each as a tuple (quaternion, angular_velocity)
    """
    # Extract quaternion and angular velocity from mean
    q_mean = mean[0]  # Quaternion object
    omega_mean = mean[1]  # 3D vector for angular velocity

    # Size of covariance matrix (should be 6 for orientation tracking)
    n = covariance.shape[0]

    # Compute scaling factor for sigma points
    scaling = np.sqrt(n + kappa)

    # Compute square root of covariance matrix using scipy's sqrtm
    S = linalg.sqrtm(covariance) * scaling  # Square root of covariance

    # List to store sigma points
    sigma_points = []

    # INCORPORATE Q NOISE (UPDATED NOISE BIG NOISE)
    # AROUND 6X6 DIAG W ~10

    # Generate 2n sigma points using positive and negative square root columns
    for i in range(n):
        # Extract i-th column of S
        S_i = S[:, i]

        # Split S_i into orientation and angular velocity parts
        # First 3 elements correspond to orientation error
        # Last 3 elements correspond to angular velocity error
        delta_orientation = S_i[:3]
        delta_omega = S_i[3:]

        # Convert orientation error to quaternion representation
        # This uses axis-angle representation: the orientation error is treated as
        # a rotation vector (axis * angle)
        # Create new quaternion instances and call from_axis_angle on them
        q_delta_pos = Quaternion()
        q_delta_pos.from_axis_angle(delta_orientation)

        q_delta_neg = Quaternion()
        q_delta_neg.from_axis_angle(-delta_orientation)

        # Create positive sigma point:
        # For quaternion: multiply mean quaternion by delta quaternion
        # For angular velocity: add delta to mean
        q_pos = q_mean * q_delta_pos
        omega_pos = omega_mean + delta_omega
        sigma_points.append((q_pos, omega_pos))

        # Create negative sigma point:
        # For quaternion: multiply mean quaternion by negative delta quaternion
        # For angular velocity: subtract delta from mean
        q_neg = q_mean * q_delta_neg
        omega_neg = omega_mean - delta_omega
        sigma_points.append((q_neg, omega_neg))

    return sigma_points

'''
def sigma_points_gen_UT(x_k_k, P_k_k, Q, dt):
    n =6
    Sigma_X = np.zeros((7,2*n))
    Sigma_Y = np.zeros((7,2*n))
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
        # unscented transform
        q_d = Quaternion()
        angle_d = Sigma_X[4:7,j] * dt
        q_d.from_axis_angle(angle_d)
        q_Y = q_X * q_d
        q_Y.normalize()
        Sigma_Y[0:4,j] = q_Y.q
        Sigma_Y[4:7,j] = Sigma_X[4:7,j]
    return Sigma_X, Sigma_Y
'''



def ukf_prediction_step(prev_mean, mean, covariance, dt, process_noise):
    # Scale process noise by time step
    scaled_process_noise = process_noise * dt

    # Add process noise to current covariance
    augmented_covariance = covariance + scaled_process_noise

    # Generate sigma points from current state and augmented covariance
    sigma_points = generate_sigma_points(mean, augmented_covariance)

    # Propagate sigma points through dynamics model
    propagated_sigma_points = []
    for sigma_point in sigma_points:
        propagated_point = propagate_dynamics(sigma_point, dt)
        propagated_sigma_points.append(propagated_point)

    # Extract quaternions and angular velocities
    q_list = [point[0] for point in propagated_sigma_points]
    omega_list = [point[1] for point in propagated_sigma_points]

    # Turn into np array so we can find mean easily
    omega_array = np.array(omega_list)
    # Use gradient descent method to get mean, also get the covariance
    q_mean, orientation_covariance = quaternion_average(q_list, prev_mean)

    # Compute mean of angular velocities (standard Euclidean mean)
    omega_mean = np.mean(omega_array, axis=0)

    # Compute angular velocity covariance (standard covariance calculation)
    n = len(sigma_points)
    omega_covariance = np.zeros((3, 3))
    for omega in omega_list:
        diff = (omega - omega_mean).reshape(-1, 1)  # Column vector
        omega_covariance += np.dot(diff, diff.T)
    omega_covariance = omega_covariance / n

    # Compute cross-covariance between orientation and angular velocity
    # !MAY NOT NEED TO DO THIS?
    # !COMBINE TOGETHER POTENTIALLY IF THINGS DO NOT WORK
    cross_covariance = np.zeros((3, 3))
    for i in range(n):
        # Get orientation error vector for this sigma point
        q_error = q_list[i] * q_mean.inv()
        e_i = q_error.axis_angle()

        # Get angular velocity error
        omega_diff = omega_list[i] - omega_mean

        # Compute contribution to cross-covariance
        cross_covariance += np.outer(e_i, omega_diff)
    cross_covariance = cross_covariance / n

    # Assemble full covariance matrix
    predicted_covariance = np.zeros((6, 6))
    predicted_covariance[0:3, 0:3] = orientation_covariance
    predicted_covariance[3:6, 3:6] = omega_covariance
    predicted_covariance[0:3, 3:6] = cross_covariance
    predicted_covariance[3:6, 0:3] = cross_covariance.T

    # Return predicted state and covariance
    predicted_mean = (q_mean, omega_mean)
    return predicted_mean, predicted_covariance


def propagate_dynamics(state, dt):
    """
    Propagate state through dynamics model.

    Args:
        state: Tuple of (quaternion, angular_velocity)
        dt: Time step (seconds)

    Returns:
        new_state: Propagated state
    """
    q, omega = state

    # Quaternion kinematics: dq/dt = 0.5 * q * [0, omega]
    # First, create a quaternion representing the angular velocity
    omega_quat = Quaternion(0, [omega[0], omega[1], omega[2]])

    # CHECK IF THIS PART IS CORRECT ESPECIALLY HARD!!!!!
    # Compute quaternion derivative
    q_omega = q * omega_quat

    # Manually multiply the components by 0.5
    q_dot_scalar = q_omega.scalar() * 0.5
    q_dot_vec = q_omega.vec() * 0.5

    # q_new = q + q_dot * dt
    q_new = Quaternion(
        q.scalar() + q_dot_scalar * dt,
        q.vec() + q_dot_vec * dt
    )

    # Normalize
    q_new.normalize()

    # Keep same?
    omega_new = omega

    return (q_new, omega_new)


def quaternion_average(quaternions, initial_q=None, max_iterations=100, tolerance=1e-8):
    """
    Compute the mean of a set of quaternions using gradient descent.
    Implements algorithm from section 3.4 of Kraft's paper.

    Args:
        quaternions: List of quaternions
        initial_q: Optional initial mean quaternion (defaults to first quaternion if None)
        max_iterations: Maximum number of iterations for convergence
        tolerance: Convergence threshold

    Returns:
        mean_quaternion: Average quaternion
    """
    n = len(quaternions)

    # Initialize mean quaternion with initial_q if provided, otherwise use first quaternion
    #q_mean = initial_q[0] if initial_q is not None else quaternions[0]
    q_mean = quaternions[0]

    # Initialize error vector matrix E (3 x n)
    E = np.zeros((3, n))

    for iteration in range(max_iterations):
        # Get errors for each quaternion
        for i in range(n):
            # Get diff between quaternion and mean
            q_error = quaternions[i] * q_mean.inv()

            # ADDED IN THIS NORMALIZATION IS IT BAD THAT THE DIFFERENCE IS CAUSING SCALAR GREATER THAN 1?
            q_error.normalize()

            # Convert to axis-angle representation (3D vector)
            E[:, i] = q_error.axis_angle()

        # Compute mean
        # take mean for x,y,z (rows)
        e_bar = np.mean(E, axis=1)

        # Step (iii): Check for convergence
        if np.linalg.norm(e_bar) < tolerance:
            break

        # Create quaternion from mean error vector
        delta_q = Quaternion()
        delta_q.from_axis_angle(e_bar)

        # Update mean quaternion: q_mean = delta_q * q_mean according to equation 53
        q_mean = delta_q * q_mean
        q_mean.normalize()

    # Step (iv): Calculate covariance of the error vectors after convergence
    orientation_covariance = np.zeros((3, 3))
    for i in range(n):
        e_i = E[:, i].reshape(-1, 1)  # Column vector
        orientation_covariance += np.dot(e_i, e_i.T)
    orientation_covariance /= n  # Divide by n

    return q_mean, orientation_covariance


if __name__ == "__main__":
    new_accel, gyro, imu_ts, vicon_rot, vicon_ts = utils.load_data(
        data_folder='hw2_p2_data', data_num=3)

    # Calibrate sensors
    accel_bias = np.array([512.16432647, 500.57738536, 502.43180196])
    accel_sensitivity = np.array([36.25710795, 34.8814697,  34.27137708])
    gyro_bias = np.array([368.05326399, 372.06745886, 375.34814997])
    gyro_sensitivity = np.array([180, 220, 220])

    # Apply calibration
    accel_calibrated = accel.calibrate_accel(new_accel, accel_bias,
                                             accel_sensitivity)
    gyro_calibrated = gyroscope.calibrate_gyro(gyro, gyro_bias,
                                               gyro_sensitivity)

    # Initialize UKF
    state, cov, process_noise, measurement_noise = initialize_ukf()

    # Create containers for results
    num_steps = len(imu_ts)
    quaternions = []
    angular_velocities = []
    covariances = []

    # Run UKF for each time step
    current_state = state
    current_cov = cov
    prev_state = None

    quaternions.append(current_state[0])
    angular_velocities.append(current_state[1])
    covariances.append(current_cov)

    for i in range(1, num_steps, 3):  # Step by 3
        # 3 time step diff
        if i + 3 < num_steps:
            # Calculate time step across multiple measurements
            dt = imu_ts[i + 3] - imu_ts[i - 1]

            # Get current measurement (either use the latest or average multiple measurements)
            accel_measurement = accel_calibrated[:, i + 3]
            gyro_measurement = gyro_calibrated[:, i + 3]
        else:
            # Handle the last few measurements that don't make a full group of 3
            dt = imu_ts[-1] - imu_ts[i - 1]
            accel_measurement = accel_calibrated[:, -1]
            gyro_measurement = gyro_calibrated[:, -1]

        # Get current measurement
        measurement = np.concatenate([accel_measurement, gyro_measurement])

        # prop dynamics
        predicted_state, predicted_cov = ukf_prediction_step(
            prev_state, current_state, current_cov, dt, process_noise)

        # measurement update
        current_state, current_cov = ukf_update_step(
            predicted_state, predicted_cov, measurement, measurement_noise)
        prev_state = current_state
        # Store results
        quaternions.append(current_state[0])
        angular_velocities.append(current_state[1])
        covariances.append(current_cov)

    # Simple plotting for UKF results

    euler_angles = np.array([q.euler_angles() for q in quaternions])

    # Get Vicon ground truth
    vicon_euler_angles = []
    for i in range(vicon_rot.shape[2]):
        q_vicon = Quaternion()
        q_vicon.from_rotm(vicon_rot[:, :, i])
        vicon_euler_angles.append(q_vicon.euler_angles())
    vicon_euler_angles = np.array(vicon_euler_angles)

    # Create timestamps that match our steps
    ukf_timestamps = []
    for i in range(0, num_steps, 3):
        ukf_timestamps.append(imu_ts[i])
    ukf_timestamps = ukf_timestamps[:len(quaternions)]

    # Make sure data lengths match
    euler_angles = euler_angles[:len(ukf_timestamps)]

    # Plot Euler angles
    plt.figure(figsize=(12, 9))

    # Roll
    plt.subplot(3, 1, 1)
    plt.plot(ukf_timestamps, euler_angles[:, 0], 'b-', label='UKF')
    plt.plot(vicon_ts.flatten(), vicon_euler_angles[:, 0], 'r--',
             label='Vicon')
    plt.title('Roll')
    plt.legend()
    plt.grid(True)

    # Pitch
    plt.subplot(3, 1, 2)
    plt.plot(ukf_timestamps, euler_angles[:, 1], 'b-', label='UKF')
    plt.plot(vicon_ts.flatten(), vicon_euler_angles[:, 1], 'r--',
             label='Vicon')
    plt.title('Pitch')
    plt.legend()
    plt.grid(True)

    # Yaw
    plt.subplot(3, 1, 3)
    plt.plot(ukf_timestamps, euler_angles[:, 2], 'b-', label='UKF')
    plt.plot(vicon_ts.flatten(), vicon_euler_angles[:, 2], 'r--',
             label='Vicon')
    plt.title('Yaw')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ukf_results.png')
    plt.show()
