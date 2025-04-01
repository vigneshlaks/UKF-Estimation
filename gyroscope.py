import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import utils
import matplotlib.pyplot as plt
import scipy.optimize as opt


def compute_vicon_angular_velocity(vicon_rot, vicon_ts):
    """Compute angular velocity from Vicon rotation matrices with robust outlier handling."""
    T = vicon_ts.shape[0]
    angular_velocity = np.zeros((3, T - 1))
    dt = np.diff(vicon_ts)
    mid_timestamps = (vicon_ts[:-1] + vicon_ts[1:]) / 2

    for i in range(T - 1):
        # get rotations
        R1 = vicon_rot[:, :, i]
        R2 = vicon_rot[:, :, i + 1]

        # find rotation
        R_rel = np.matmul(R2, R1.T)
        rot = R.from_matrix(R_rel)

        # turn into vector
        rotvec = rot.as_rotvec()

        # divide by time pass
        angular_velocity[:, i] = rotvec / dt[i]

    # Identify outliers using a moving median filter approach
    window_size = 5  # Adjust window size as needed

    # Handle each axis separately
    for axis in range(3):
        # Calculate moving median
        # pad so that size stays the same
        padded = np.pad(angular_velocity[axis],
                        (window_size // 2, window_size // 2), mode='edge')
        moving_median = np.array([np.median(padded[i:i + window_size])
                                  for i in range(len(angular_velocity[axis]))])

        # Calculate deviation from moving median
        deviation = np.abs(angular_velocity[axis] - moving_median)

        # Define threshold as multiple of median absolute deviation (MAD)
        mad = np.median(deviation)
        threshold = 3.0 * mad  # Adjust multiplier as needed

        # Identify outliers
        outlier_mask = deviation > threshold

        # Replace outliers with interpolated values
        if np.any(outlier_mask):
            # Create interpolation indices
            valid_indices = np.where(~outlier_mask)[0]
            outlier_indices = np.where(outlier_mask)[0]

            # Only interpolate if we have enough valid points
            if len(valid_indices) > 1:
                # Use linear interpolation to fill outliers
                valid_values = angular_velocity[axis, valid_indices]

                # Create interpolation function
                from scipy.interpolate import interp1d
                f = interp1d(valid_indices, valid_values,
                             bounds_error=False, fill_value='extrapolate')

                # Replace outlier values
                angular_velocity[axis, outlier_indices] = f(outlier_indices)
            else:
                # If not enough valid points, use median
                angular_velocity[axis, outlier_indices] = np.median(
                    angular_velocity[axis])

    return angular_velocity, mid_timestamps


def calibrate_gyro(gyro, gyro_bias, gyro_sensitivity):
    """
    Calibrate gyroscope data with axis reordering.
    The gyroscope axes need to be reordered to match the coordinate system.
    """
    adc_to_mV = 3300 / 1023

    # First calibrate the values
    calibrated = (gyro - gyro_bias[:, None]) * (
            adc_to_mV / gyro_sensitivity[:, None])

    # Create a new array for the reordered data
    reordered = np.zeros_like(calibrated)

    # reordering axes
    reordered[0, :] = calibrated[1, :]
    reordered[1, :] = calibrated[2, :]
    reordered[2, :] = calibrated[0, :]

    return reordered


def calculate_calibration_error(calib_gyro, imu_ts, vicon_omega,
                                vicon_omega_ts, yaw_weight=1):
    """Calculate mean absolute error between calibrated gyro and Vicon data with emphasis on yaw."""
    # Find common time range
    imu_start, imu_end = imu_ts[0], imu_ts[-1]
    vicon_start, vicon_end = vicon_omega_ts[0], vicon_omega_ts[-1]

    # Find overlapping time range
    start_time = max(imu_start, vicon_start)
    end_time = min(imu_end, vicon_end)

    # Filter timestamps to common range
    imu_mask = (imu_ts >= start_time) & (imu_ts <= end_time)
    vicon_mask = (vicon_omega_ts >= start_time) & (vicon_omega_ts <= end_time)

    if np.sum(imu_mask) < 10 or np.sum(vicon_mask) < 10:
        return np.inf  # Not enough overlap

    # Interpolate Vicon data to IMU timestamps in common range
    errors = []
    weights = [1.0, 1.0, yaw_weight]  # Assuming Z-axis (index 2) is yaw

    for axis in range(3):
        f_interp = interp1d(vicon_omega_ts[vicon_mask],
                            vicon_omega[axis, vicon_mask],
                            bounds_error=False, fill_value=np.nan)
        vicon_aligned = f_interp(imu_ts[imu_mask])

        # Calculate absolute difference, ignoring outliers
        diff = np.abs(calib_gyro[axis, imu_mask] - vicon_aligned)
        valid_mask = (~np.isnan(diff)) & (
                diff <= 1.0)  # Ignore differences > 1.0 rad/s

        if np.sum(valid_mask) < 0.5 * len(
                vicon_aligned):  # At least 50% valid points
            return np.inf

        # Apply weight to this axis
        axis_error = np.nanmean(diff[valid_mask]) * weights[axis]
        errors.append(axis_error)

    # Calculate weighted average of errors
    return np.sum(errors) / np.sum(weights)


def find_best_calib_gyro_nm(gyro_raw, imu_ts, vicon_rot, vicon_ts):
    """
    Find the optimal gyroscope calibration parameters using the Nelder-Mead method.
    The parameter vector x is defined as:
        x = [bias_x, bias_y, bias_z, sens_x, sens_y, sens_z]
    """
    # Pre-compute Vicon angular velocity and mid timestamps
    vicon_omega, vicon_omega_ts = compute_vicon_angular_velocity(vicon_rot, vicon_ts)

    def objective(x):
        # Split parameter vector into bias and sensitivity components
        gyro_bias = np.array(x[:3])
        gyro_sens = np.array(x[3:])
        # Calibrate the gyro data using these parameters
        calib = calibrate_gyro(gyro_raw, gyro_bias, gyro_sens)
        # Compute and return the calibration error
        error = calculate_calibration_error(calib, imu_ts, vicon_omega, vicon_omega_ts)
        return error

    # Initial guess based on provided calibration ranges:
    # Gyroscope: bias ~ 350, sensitivity ~ 200
    x0 = np.array([350, 350, 350, 200, 200, 200])

    # Run the Nelder-Mead optimization
    #res = opt.minimize(objective, x0, method='Nelder-Mead')

    bounds = [(300, 400), (300, 400), (300, 400),  # bias bounds
              (180, 220), (180, 220), (180, 220)]  # sensitivity bounds

    res = opt.minimize(objective, x0, method='L-BFGS-B',
                       bounds=bounds)

    best_bias = res.x[:3]
    best_sens = res.x[3:]
    print("Nelder-Mead Optimization Result:")
    print(f"Best Bias: {best_bias}")
    print(f"Best Sensitivity: {best_sens}")
    print(f"Final Error: {res.fun:.4f}")

    # Compute the final calibrated gyroscope data using the optimal parameters
    best_calib = calibrate_gyro(gyro_raw, best_bias, best_sens)
    return best_calib

def plot_gyro_calibration_nm(gyro_raw, imu_ts, vicon_rot, vicon_ts):
    """Plot calibrated gyro data against Vicon angular velocity using Nelder-Mead optimization."""
    # Get calibrated gyro data via Nelder-Mead optimization
    calib_gyro = find_best_calib_gyro_nm(gyro_raw, imu_ts, vicon_rot, vicon_ts)

    # Get Vicon angular velocity
    vicon_omega, vicon_omega_ts = compute_vicon_angular_velocity(vicon_rot, vicon_ts)

    # Create plot
    axis_labels = ['X', 'Y', 'Z']
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(imu_ts, calib_gyro[i, :], label='Calibrated Gyro', linewidth=2)
        plt.plot(vicon_omega_ts, vicon_omega[i, :],
                 label='Vicon Angular Velocity', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title(f"Angular Velocity - {axis_labels[i]} Axis")
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load data
    new_accel, gyro, imu_ts, vicon_rot, vicon_ts = utils.load_data(
        data_folder='hw2_p2_data', data_num=3)

    print(
        f"Loaded IMU data ({imu_ts.shape[0]} samples) and Vicon data ({vicon_ts.shape[0]} samples)")

    # Run calibration and plotting
    plot_gyro_calibration_nm(gyro, imu_ts, vicon_rot, vicon_ts)