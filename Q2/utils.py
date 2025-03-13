import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_data(data_folder='hw2_p2_data', data_num=1):
    """
    Loads IMU and Vicon data
    """
    # Load IMU
    imu_data = scipy.io.loadmat(f'{data_folder}/imu/imuRaw{data_num}.mat')
    accel = imu_data['vals'][0:3, :]
    gyro = imu_data['vals'][3:6, :]
    imu_ts = imu_data['ts'].flatten()

    # Load Vicon
    vicon_data = scipy.io.loadmat(
        f'{data_folder}/vicon/viconRot{data_num}.mat')
    vicon_rot = vicon_data['rots']

    vicon_ts = vicon_data['ts'].flatten()

    return accel, gyro, imu_ts, vicon_rot, vicon_ts

def extract_vicon_values(vicon_rot):
    N_vicon = vicon_rot.shape[2]
    euler_vicon = []
    for i in range(N_vicon):
        R_mat = vicon_rot[:, :, i]
        r = R.from_matrix(R_mat)
        euler = r.as_euler('xyz', degrees=False)  # returns roll, pitch, yaw
        euler_vicon.append(euler)
    euler_vicon = np.array(euler_vicon).T  # shape (3, N_vicon)
    roll_vicon = euler_vicon[0, :]
    pitch_vicon = euler_vicon[1, :]
    yaw_vicon = euler_vicon[2, :]

    return roll_vicon, pitch_vicon, yaw_vicon


def calculate_alignment_error(measured_data, reference_data, time_shift=15,
                              weights=None):
    """
    Calculate mean absolute error between measured data and reference data with time alignment.
    Can be used for any pair of time series data arrays (accelerometer, gyroscope, etc.)

    Args:
        measured_data: Data array from sensor (N x D) where N is number of samples, D is dimensions
        reference_data: Reference data array (M x D) where M is number of samples, D is dimensions
        time_shift: Number of samples to shift reference data (default=15)
        weights: Optional weights for each dimension (array of length D)

    Returns:
        total_error: Combined weighted error across all dimensions
        dimension_errors: Individual errors for each dimension
    """
    # Check if inputs are 1D arrays and convert to 2D if needed
    if measured_data.ndim == 1:
        measured_data = measured_data.reshape(1, -1)
    if reference_data.ndim == 1:
        reference_data = reference_data.reshape(1, -1)

    # Get dimensions
    D = measured_data.shape[0]  # Number of dimensions/axes

    # Set default weights if not provided
    if weights is None:
        weights = np.ones(D)

    # Apply time shift (shift reference data to the left)
    if time_shift > 0 and time_shift < reference_data.shape[1]:
        reference_data_shifted = reference_data[:, time_shift:]
    else:
        reference_data_shifted = reference_data

    # Find the minimum length after shifting
    min_samples = min(measured_data.shape[1], reference_data_shifted.shape[1])

    # Calculate errors for each dimension
    dimension_errors = np.zeros(D)
    for i in range(D):
        dimension_errors[i] = np.mean(
            np.abs(measured_data[i, :min_samples] - reference_data_shifted[i,
                                                    :min_samples]))

    # Calculate weighted total error
    total_error = np.sum(dimension_errors * weights)

    return total_error, dimension_errors

