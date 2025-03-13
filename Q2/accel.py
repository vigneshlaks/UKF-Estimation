import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, \
    extract_vicon_values  # adjust import path as needed
from scipy.spatial.transform import Rotation as R

# Global variables for Vicon ground truth angles (set after data loading)
ROLL_VICON, PITCH_VICON, YAW_VICON = None, None, None


def calibrate_accel(accel, accel_bias, accel_sensitivity):
    adc_to_mV = 3300 / 1023
    # Calibrate accelerometer: convert raw ADC values to physical units.
    calib_accel = (accel - accel_bias[:, None]) * (
                adc_to_mV / accel_sensitivity[:, None])
    # Flip X and Y axes if needed due to device design.
    calib_accel[0, :] = -calib_accel[0, :]
    calib_accel[1, :] = -calib_accel[1, :]
    return calib_accel


def get_roll_pitch_accel(calib_accel):
    # Compute roll and pitch from calibrated accelerometer data.
    ax = calib_accel[0, :]
    ay = calib_accel[1, :]
    az = calib_accel[2, :]
    eps = 1e-6
    roll_accel = np.arctan2(ay, az + eps)
    pitch_accel = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2) + eps)
    return roll_accel, pitch_accel


def mean_absolute_error(calib_accel, time_shift=15):
    """
    Calculate mean absolute error between accelerometer-derived and Vicon roll and pitch.
    The error is defined as roll_error + 5 * pitch_error.
    """
    roll_accel, pitch_accel = get_roll_pitch_accel(calib_accel)
    # Assume global Vicon angles have been set.
    roll_vicon_shifted = ROLL_VICON[time_shift:]
    pitch_vicon_shifted = PITCH_VICON[time_shift:]
    min_samples = min(len(roll_accel), len(roll_vicon_shifted))
    roll_error = np.mean(
        np.abs(roll_accel[:min_samples] - roll_vicon_shifted[:min_samples]))
    pitch_error = np.mean(
        np.abs(pitch_accel[:min_samples] - pitch_vicon_shifted[:min_samples]))
    total_error = roll_error + 5 * pitch_error
    return total_error, roll_error, pitch_error


def has_valid_rest_period(calib_accel, std_threshold=0.15, window_size=20,
                          magnitude_tolerance=0.15):
    """
    Check whether the first window of the calibrated data shows a stationary period:
    the acceleration magnitude should be close to gravity (9.81 m/s²) and have low variance.
    """
    if calib_accel.shape[1] < window_size:
        return False
    calib_accel_mag = np.sqrt(
        np.sum(calib_accel[:, :window_size] ** 2, axis=0))
    window_std = np.std(calib_accel_mag)
    if window_std < std_threshold:
        mean_mag = np.mean(calib_accel_mag)
        if abs(mean_mag - 9.81) <= magnitude_tolerance:
            return True
    return False


def find_best_calib_accel_nm(accel, time_shift=15):
    """
    Use the Nelder–Mead optimization to determine the best accelerometer calibration parameters.
    The parameter vector x is defined as:
      x = [bias_x, bias_y, bias_z, sensitivity_x, sensitivity_y, sensitivity_z]
    Returns the calibrated accelerometer data using the optimal parameters.
    """

    def objective(x):
        # Split candidate parameter vector
        accel_bias = np.array(x[:3])
        accel_sensitivity = np.array(x[3:])
        calib = calibrate_accel(accel, accel_bias, accel_sensitivity)
        # If a valid rest period is not found, penalize heavily.
        if not has_valid_rest_period(calib):
            return 1e6
        total_error, _, _ = mean_absolute_error(calib, time_shift)
        return total_error

    # Initial guess: biases near 500 and sensitivities near 35 (mV/(m/s²))
    x0 = np.array([500, 500, 500, 35, 35, 35])
    res = opt.minimize(objective, x0, method='Nelder-Mead')

    best_bias = res.x[:3]
    best_sens = res.x[3:]
    print("Nelder–Mead Accelerometer Calibration Result:")
    print(f"  Best Bias: {best_bias}")
    print(f"  Best Sensitivity: {best_sens} mV/(m/s²)")
    print(f"  Final Total Error: {res.fun:.6f}")

    best_calib = calibrate_accel(accel, best_bias, best_sens)
    return best_calib


def plot_accel_calibration(calib_accel, timestamps, time_shift=15):
    """
    Plot calibrated accelerometer data (magnitude, roll, and pitch) alongside Vicon ground truth.
    """
    accel_mag = np.sqrt(np.sum(calib_accel ** 2, axis=0))
    roll_acc, pitch_acc = get_roll_pitch_accel(calib_accel)
    roll_vicon_shifted = ROLL_VICON[time_shift:]
    pitch_vicon_shifted = PITCH_VICON[time_shift:]
    min_samples = min(len(roll_acc), len(roll_vicon_shifted))
    roll_acc_aligned = roll_acc[:min_samples]
    pitch_acc_aligned = pitch_acc[:min_samples]
    accel_mag_aligned = accel_mag[:min_samples]
    t_common = timestamps[:min_samples]

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(t_common, accel_mag_aligned, label="Accel Magnitude")
    plt.axhline(y=9.81, color='r', linestyle='--',
                label="Expected gravity (9.81 m/s²)")
    plt.title("Acceleration Magnitude (should be ~9.81 m/s² at rest)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t_common, roll_acc_aligned, label="Accel-derived Roll")
    plt.plot(t_common, roll_vicon_shifted[:min_samples],
             label="Vicon Roll (shifted)", linestyle="--")
    plt.title("Roll Comparison (with time alignment)")
    plt.xlabel("Time (s)")
    plt.ylabel("Roll (radians)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t_common, pitch_acc_aligned, label="Accel-derived Pitch")
    plt.plot(t_common, pitch_vicon_shifted[:min_samples],
             label="Vicon Pitch (shifted)", linestyle="--")
    plt.title("Pitch Comparison (with time alignment)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (radians)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load data (adjust data_folder and data_num as needed)
    accel, gyro, timestamps, vicon_rot, vicon_timestamps = load_data(
        data_folder='hw2_p2_data', data_num=1)
    # Create Vicon timestamps assuming uniform sampling across the Vicon file.
    vicon_ts = np.linspace(timestamps[0], timestamps[-1], vicon_rot.shape[2])
    # Extract Vicon Euler angles and set global variables for error functions.
    ROLL_VICON, PITCH_VICON, YAW_VICON = extract_vicon_values(vicon_rot)

    print(f"Loaded {timestamps.shape[0]} IMU timestamps")
    print(f"IMU Accelerometer shape: {accel.shape}")
    print(f"IMU Gyroscope shape: {gyro.shape}")
    print(f"Vicon Rotation shape: {vicon_rot.shape}")

    # Calibrate accelerometer using Nelder–Mead optimization.
    calib_accel = find_best_calib_accel_nm(accel, time_shift=15)

    # Plot accelerometer calibration results.
    plot_accel_calibration(calib_accel, timestamps, time_shift=15)