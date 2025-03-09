import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import itertools
from tqdm import tqdm  # Optional, for progress bar


def load_data(data_folder='hw2_p2_data', data_num=1):
    """
    Loads IMU and Vicon data
    """
    # Load IMU
    imu_data = scipy.io.loadmat(f'{data_folder}/imu/imuRaw{data_num}.mat')
    accel = imu_data['vals'][0:3, :]  # Accelerometer readings
    gyro = imu_data['vals'][3:6, :]  # Gyroscope readings
    timestamps = imu_data['ts'].flatten()  # Timestamps

    # Load Vicon
    vicon_data = scipy.io.loadmat(
        f'{data_folder}/vicon/viconRot{data_num}.mat')
    vicon_rot = vicon_data['rots']

    return accel, gyro, timestamps, vicon_rot


def calculate_orientation(accel):
    """
    Calculates roll and pitch from accelerometer data
    """
    ax = accel[0]
    ay = accel[1]
    az = accel[2]

    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

    return roll, pitch


def extract_vicon_orientation(vicon_rot):
    """
    Extracts roll, pitch, yaw from Vicon rotation matrices
    """
    N = vicon_rot.shape[2]
    euler = np.zeros((3, N))

    for i in range(N):
        r = R.from_matrix(vicon_rot[:, :, i])
        euler[:, i] = r.as_euler('xyz', degrees=False)

    return euler[0], euler[1], euler[2]  # roll, pitch, yaw


def calculate_angular_velocity(vicon_rot, timestamps):
    """
    Calculates angular velocity from Vicon rotation matrices
    """
    N = vicon_rot.shape[2]
    angular_velocity = np.zeros((3, N - 1))

    for i in range(N - 1):
        R1 = vicon_rot[:, :, i]
        R2 = vicon_rot[:, :, i + 1]

        # Get rotation between frames
        dR = R2 @ R1.T

        # Convert to axis-angle representation
        r = R.from_matrix(dR)
        rotvec = r.as_rotvec()

        # Divide by time step to get angular velocity
        dt = timestamps[i + 1] - timestamps[i]
        angular_velocity[:, i] = rotvec / dt

    return angular_velocity


def apply_calibration(accel, gyro, accel_params, gyro_params):
    """
    Apply calibration parameters to raw sensor data

    Args:
        accel: Raw accelerometer data (3, N)
        gyro: Raw gyroscope data (3, N)
        accel_params: (bias, sensitivity) for accelerometer
        gyro_params: (bias, sensitivity) for gyroscope

    Returns:
        calib_accel: Calibrated accelerometer data (3, N)
        calib_gyro: Calibrated gyroscope data (3, N)
    """
    accel_bias, accel_sensitivity = accel_params
    gyro_bias, gyro_sensitivity = gyro_params

    adc_to_mV = 3300 / 1023

    # Calibrate accelerometer
    calib_accel = np.zeros_like(accel)
    for i in range(3):
        calib_accel[i] = (accel[i] - accel_bias[i]) * (
                    adc_to_mV / accel_sensitivity[i]) * 9.81

    # Apply sign flip for x and y axes
    calib_accel[0] = -calib_accel[0]
    calib_accel[1] = -calib_accel[1]

    # Calibrate gyroscope
    calib_gyro = np.zeros_like(gyro)
    for i in range(3):
        calib_gyro[i] = (gyro[i] - gyro_bias[i]) * (
                    adc_to_mV / gyro_sensitivity[i])

    return calib_accel, calib_gyro


def calculate_error(accel, gyro, accel_params, gyro_params, vicon_rot,
                    timestamps):
    """
    Calculates error metrics between calibrated data and Vicon ground truth
    """
    # Apply calibration
    calib_accel, calib_gyro = apply_calibration(accel, gyro, accel_params,
                                                gyro_params)

    # Get IMU orientation
    roll_imu, pitch_imu = calculate_orientation(calib_accel)

    # Get Vicon orientation
    roll_vicon, pitch_vicon, _ = extract_vicon_orientation(vicon_rot)

    # Calculate orientation error
    min_samples = min(len(roll_imu), len(roll_vicon))
    roll_error = np.mean(
        (roll_imu[:min_samples] - roll_vicon[:min_samples]) ** 2)
    pitch_error = np.mean(
        (pitch_imu[:min_samples] - pitch_vicon[:min_samples]) ** 2)

    # Calculate acceleration magnitude error
    accel_mag = np.sqrt(np.sum(calib_accel ** 2, axis=0))
    mag_error = np.mean((accel_mag - 9.81) ** 2)

    # Get Vicon angular velocity for gyro comparison
    vicon_angular_vel = calculate_angular_velocity(vicon_rot, timestamps)

    # Calculate gyro error
    min_samples = min(calib_gyro.shape[1], vicon_angular_vel.shape[1])
    gyro_error = np.mean((calib_gyro[:, :min_samples] - vicon_angular_vel[:,
                                                        :min_samples]) ** 2)

    # Combined error (weighted sum)
    total_error = mag_error + 2 * roll_error + 2 * pitch_error + gyro_error

    return total_error, mag_error, roll_error, pitch_error, gyro_error


def grid_search_calibration(accel, gyro, vicon_rot, timestamps,
                            coarse_search=True):
    """
    Performs grid search to find optimal calibration parameters.

    Args:
        accel: Raw accelerometer data
        gyro: Raw gyroscope data
        vicon_rot: Vicon rotation matrices
        timestamps: Timestamps
        coarse_search: If True, performs a coarse search first, then refines around best parameters

    Returns:
        accel_params: (bias, sensitivity) for accelerometer
        gyro_params: (bias, sensitivity) for gyroscope
    """
    # Define search ranges based on the problem statement
    # For coarse search, use wider steps
    if coarse_search:
        # Accelerometer
        accel_bias_values = [470, 490, 510, 530, 550]  # Around 500
        accel_sens_values = [25, 30, 35, 40, 45, 50]  # 25-50 mV/(m/s²)

        # Gyroscope
        gyro_bias_values = [320, 335, 350, 365, 380]  # Around 350
        gyro_sens_values = [180, 190, 200, 210, 220]  # Around 200 mV/(rad/s)
    else:
        # Use pre-defined fine-grained parameters (defined later)
        pass

    # Initialize for best parameters
    best_error = float('inf')
    best_accel_params = None
    best_gyro_params = None
    best_error_components = None

    # Start with simplified search - assume same bias and sensitivity for all axes
    print("Starting coarse grid search...")

    # Total combinations to search
    total_combinations = len(accel_bias_values) * len(accel_sens_values) * len(
        gyro_bias_values) * len(gyro_sens_values)
    print(f"Searching through {total_combinations} parameter combinations...")

    # Create all parameter combinations
    param_combinations = list(itertools.product(
        accel_bias_values, accel_sens_values, gyro_bias_values,
        gyro_sens_values
    ))

    # Search through all combinations
    for params in tqdm(param_combinations):
        ab, asens, gb, gsens = params

        # Create parameter arrays (same value for all axes initially)
        accel_bias = np.array([ab, ab, ab])
        accel_sensitivity = np.array([asens, asens, asens])
        gyro_bias = np.array([gb, gb, gb])
        gyro_sensitivity = np.array([gsens, gsens, gsens])

        # Calculate error
        error, mag_error, roll_error, pitch_error, gyro_error = calculate_error(
            accel, gyro,
            (accel_bias, accel_sensitivity),
            (gyro_bias, gyro_sensitivity),
            vicon_rot, timestamps
        )

        # Update best parameters if this is better
        if error < best_error:
            best_error = error
            best_accel_params = (accel_bias.copy(), accel_sensitivity.copy())
            best_gyro_params = (gyro_bias.copy(), gyro_sensitivity.copy())
            best_error_components = (
            mag_error, roll_error, pitch_error, gyro_error)

    # Report best parameters from coarse search
    print("\nBest parameters from coarse search:")
    print(f"Accelerometer bias: {best_accel_params[0]}")
    print(f"Accelerometer sensitivity: {best_accel_params[1]} mV/(m/s²)")
    print(f"Gyroscope bias: {best_gyro_params[0]}")
    print(f"Gyroscope sensitivity: {best_gyro_params[1]} mV/(rad/s)")
    print(f"Total error: {best_error:.6f}")
    print(
        f"Error components (mag, roll, pitch, gyro): {best_error_components}")

    # If requested, perform fine-grained search around best parameters
    if coarse_search:
        print("\nPerforming fine-grained search around best parameters...")

        # Create fine-grained search ranges around best parameters
        # For each parameter, search ±5% around the best value
        ab_best, as_best = best_accel_params
        gb_best, gs_best = best_gyro_params

        # Define fine-grained search per axis
        fine_search_ranges = []

        # For each accelerometer axis
        for i in range(3):
            # Bias: search ±10 around best
            ab_range = np.linspace(ab_best[i] - 10, ab_best[i] + 10, 5)
            # Sensitivity: search ±5 around best
            as_range = np.linspace(as_best[i] - 5, as_best[i] + 5, 5)
            fine_search_ranges.append((ab_range, as_range))

        # For each gyroscope axis
        for i in range(3):
            # Bias: search ±10 around best
            gb_range = np.linspace(gb_best[i] - 10, gb_best[i] + 10, 5)
            # Sensitivity: search ±10 around best
            gs_range = np.linspace(gs_best[i] - 10, gs_best[i] + 10, 5)
            fine_search_ranges.append((gb_range, gs_range))

        # Now search per axis to refine parameters
        for axis in range(3):  # For each accelerometer axis
            print(f"\nRefining accelerometer axis {axis}...")
            ab_range, as_range = fine_search_ranges[axis]

            for ab in ab_range:
                for asens in as_range:
                    # Update just this axis
                    accel_bias = best_accel_params[0].copy()
                    accel_sensitivity = best_accel_params[1].copy()
                    accel_bias[axis] = ab
                    accel_sensitivity[axis] = asens

                    # Calculate error
                    error, mag_error, roll_error, pitch_error, gyro_error = calculate_error(
                        accel, gyro,
                        (accel_bias, accel_sensitivity),
                        best_gyro_params,
                        vicon_rot, timestamps
                    )

                    # Update if better
                    if error < best_error:
                        best_error = error
                        best_accel_params = (
                        accel_bias.copy(), accel_sensitivity.copy())
                        best_error_components = (
                        mag_error, roll_error, pitch_error, gyro_error)

        for axis in range(3):  # For each gyroscope axis
            print(f"\nRefining gyroscope axis {axis}...")
            gb_range, gs_range = fine_search_ranges[
                axis + 3]  # +3 because gyro comes after accel

            for gb in gb_range:
                for gsens in gs_range:
                    # Update just this axis
                    gyro_bias = best_gyro_params[0].copy()
                    gyro_sensitivity = best_gyro_params[1].copy()
                    gyro_bias[axis] = gb
                    gyro_sensitivity[axis] = gsens

                    # Calculate error
                    error, mag_error, roll_error, pitch_error, gyro_error = calculate_error(
                        accel, gyro,
                        best_accel_params,
                        (gyro_bias, gyro_sensitivity),
                        vicon_rot, timestamps
                    )

                    # Update if better
                    if error < best_error:
                        best_error = error
                        best_gyro_params = (
                        gyro_bias.copy(), gyro_sensitivity.copy())
                        best_error_components = (
                        mag_error, roll_error, pitch_error, gyro_error)

    # Final best parameters
    print("\nFinal best parameters:")
    print(f"Accelerometer bias: {best_accel_params[0]}")
    print(f"Accelerometer sensitivity: {best_accel_params[1]} mV/(m/s²)")
    print(f"Gyroscope bias: {best_gyro_params[0]}")
    print(f"Gyroscope sensitivity: {best_gyro_params[1]} mV/(rad/s)")
    print(f"Total error: {best_error:.6f}")
    print(
        f"Error components (mag, roll, pitch, gyro): {best_error_components}")

    return best_accel_params, best_gyro_params


def plot_results(accel, gyro, accel_params, gyro_params, vicon_rot,
                 timestamps):
    """
    Plots the calibration results against Vicon ground truth
    """
    # Apply calibration
    calib_accel, calib_gyro = apply_calibration(accel, gyro, accel_params,
                                                gyro_params)

    # Calculate orientation from IMU
    roll_imu, pitch_imu = calculate_orientation(calib_accel)

    # Get Vicon orientation
    roll_vicon, pitch_vicon, yaw_vicon = extract_vicon_orientation(vicon_rot)

    # Calculate acceleration magnitude
    accel_mag = np.sqrt(np.sum(calib_accel ** 2, axis=0))

    # Get angular velocity from Vicon
    vicon_angular_vel = calculate_angular_velocity(vicon_rot, timestamps)

    # Create plots
    plt.figure(figsize=(15, 12))

    # Plot 1: Acceleration magnitude
    plt.subplot(3, 2, 1)
    plt.plot(timestamps[:len(accel_mag)], accel_mag)
    plt.axhline(y=9.81, color='r', linestyle='--', label='9.81 m/s²')
    plt.title('Acceleration Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Roll angle comparison
    plt.subplot(3, 2, 3)
    min_samples = min(len(roll_imu), len(roll_vicon))
    plt.plot(timestamps[:min_samples], roll_imu[:min_samples], label='IMU')
    plt.plot(timestamps[:min_samples], roll_vicon[:min_samples], 'r--',
             label='Vicon')
    plt.title('Roll Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Roll (rad)')
    plt.legend()
    plt.grid(True)

    # Plot 3: Pitch angle comparison
    plt.subplot(3, 2, 5)
    plt.plot(timestamps[:min_samples], pitch_imu[:min_samples], label='IMU')
    plt.plot(timestamps[:min_samples], pitch_vicon[:min_samples], 'r--',
             label='Vicon')
    plt.title('Pitch Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (rad)')
    plt.legend()
    plt.grid(True)

    # Plot 4-6: Angular velocity comparison (3 axes)
    min_samples = min(calib_gyro.shape[1], vicon_angular_vel.shape[1])

    plt.subplot(3, 2, 2)
    plt.plot(timestamps[:min_samples], calib_gyro[0, :min_samples],
             label='IMU')
    plt.plot(timestamps[:min_samples], vicon_angular_vel[0, :min_samples],
             'r--', label='Vicon')
    plt.title('Angular Velocity (X-axis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(timestamps[:min_samples], calib_gyro[1, :min_samples],
             label='IMU')
    plt.plot(timestamps[:min_samples], vicon_angular_vel[1, :min_samples],
             'r--', label='Vicon')
    plt.title('Angular Velocity (Y-axis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(timestamps[:min_samples], calib_gyro[2, :min_samples],
             label='IMU')
    plt.plot(timestamps[:min_samples], vicon_angular_vel[2, :min_samples],
             'r--', label='Vicon')
    plt.title('Angular Velocity (Z-axis)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_params(accel_params, gyro_params, filename='calibration_params.npy'):
    """
    Saves calibration parameters for future use
    """
    params = {
        'accel_bias': accel_params[0],
        'accel_sensitivity': accel_params[1],
        'gyro_bias': gyro_params[0],
        'gyro_sensitivity': gyro_params[1]
    }
    np.save(filename, params)
    print(f"Calibration parameters saved to {filename}")


def main():
    # Load data
    accel, gyro, timestamps, vicon_rot = load_data(data_folder='hw2_p2_data',
                                                   data_num=1)

    print(f"Loaded {timestamps.shape[0]} IMU timestamps")
    print(f"IMU Accelerometer shape: {accel.shape}")
    print(f"IMU Gyroscope shape: {gyro.shape}")
    print(f"Vicon Rotation shape: {vicon_rot.shape}")

    # Perform grid search calibration
    # Set coarse_search=False to skip the two-stage process if time is limited
    accel_params, gyro_params = grid_search_calibration(
        accel, gyro, vicon_rot, timestamps, coarse_search=False
    )

    # Plot results
    plot_results(accel, gyro, accel_params, gyro_params, vicon_rot, timestamps)

    # Save parameters for future use
    save_params(accel_params, gyro_params)

    return accel_params, gyro_params


if __name__ == "__main__":
    main()