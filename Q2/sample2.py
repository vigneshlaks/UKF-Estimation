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
    timestamps = imu_data['ts'].flatten()

    # Load Vicon
    vicon_data = scipy.io.loadmat(
        f'{data_folder}/vicon/viconRot{data_num}.mat')
    vicon_rot = vicon_data['rots']

    return accel, gyro, timestamps, vicon_rot

def calibrate_accel(accel, accel_bias, accel_sensitivity):

    adc_to_mV = 3300 / 1023

    # Calibrate accelerometer:
    calib_accel = (accel - accel_bias[:, None]) * (
            adc_to_mV / accel_sensitivity[:, None]) * 9.81

    # Flip due to device design or wtv
    calib_accel[0, :] = -calib_accel[0, :]
    calib_accel[1, :] = -calib_accel[1, :]

    return calib_accel

def calibrate_gyro(gyro, gyro_bias, gyro_sensitivity):
    adc_to_mV = 3300 / 1023

    return (gyro - gyro_bias[:, None]) * (
                adc_to_mV / gyro_sensitivity[:, None])

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


def get_roll_pitch_accel(calib_accel):
    # get roll and pitch from calibrated accel
    ax = calib_accel[0, :]
    ay = calib_accel[1, :]
    az = calib_accel[2, :]

    eps = 1e-6
    roll_accel = np.arctan2(ay, az + eps)
    pitch_accel = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2) + eps)

    return roll_accel, pitch_accel

def mean_absolute_error(calib_accel):
    roll_accel, pitch_accel = get_roll_pitch_accel(calib_accel)

    roll_vicon, pitch_vicon = ROLL_VICON, PITCH_VICON
    min_samples = min(len(roll_accel), len(roll_vicon))

    roll_error = np.mean(np.abs(roll_accel[:min_samples] - roll_vicon[:min_samples]))
    pitch_error = np.mean(np.abs(pitch_accel[:min_samples] - pitch_vicon[:min_samples]))

    total_error = roll_error + pitch_error

    return total_error, roll_error, pitch_error

def calculate_accel_orientation_error(calib_accel):
    """
    Calculates orientation error metrics between accelerometer and Vicon.

    Args:
        calib_accel: Calibrated accelerometer data (3, N)
        vicon_rot: Vicon rotation matrices for ground truth (3, 3, M)

    Returns:
        total_error: Combined orientation error
        roll_error: Mean squared error in roll angle
        pitch_error: Mean squared error in pitch angle
    """

    roll_accel, pitch_accel = get_roll_pitch_accel(calib_accel)

    roll_vicon, pitch_vicon = ROLL_VICON, PITCH_VICON

    # Calculate orientation errors using the minimum number of samples
    min_samples = min(len(roll_accel), len(roll_vicon))

    # Mean squared error for roll
    roll_error = np.mean(
        (roll_accel[:min_samples] - roll_vicon[:min_samples]) ** 2)

    # Mean squared error for pitch
    pitch_error = np.mean(
        (pitch_accel[:min_samples] - pitch_vicon[:min_samples]) ** 2)

    # Combined error (equal weighting)
    total_error = roll_error + pitch_error

    return total_error, roll_error, pitch_error

def find_best_calib_accel(accel):
    """
    Performs grid search to find optimal accelerometer calibration parameters.

    Args:
        accel: Raw accelerometer data (3, N)
        vicon_rot: Vicon rotation matrices for ground truth (3, 3, M)

    Returns:
        best_bias: Optimal bias values (3,)
        best_sensitivity: Optimal sensitivity values (3,)
    """
    # Define search ranges based on the problem statement
    #bias_values = [470, 480, 490, 500, 510, 520, 530]  # Around 500
    #sensitivity_values = [25, 30, 35, 40, 45, 50]  # 25-50 mV/(m/s²)

    bias_values = [470, 480, 490, 500, 510, 520, 530]  # Around 500
    sensitivity_values = [25, 30, 35, 40, 45, 50]  # 25-50 mV/(m/s²)

    # Initialize for best parameters
    best_error = float('inf')
    best_bias = None
    best_sensitivity = None
    best_accel = None

    # Print search information
    total_combinations = len(bias_values) * len(sensitivity_values)
    print(
        f"Searching through {total_combinations} accelerometer parameter combinations...")

    # Loop through all parameter combinations
    for bias in bias_values:
        for sensitivity in sensitivity_values:
            if bias == 500:
                print("sadkl;hv lk  hE DJKLSHAF LKJSDHFL JKSADF ")
            # Create parameter arrays (same value for all axes)
            accel_bias = np.array([bias, bias, bias])
            accel_sensitivity = np.array(
                [sensitivity, sensitivity, sensitivity])

            # Apply calibration
            calib_accel = calibrate_accel(accel, accel_bias,
                                                  accel_sensitivity)

            # Calculate orientation error
            total_error, roll_error, pitch_error = mean_absolute_error(
                calib_accel
            )

            # Update best parameters if this is better
            if total_error < best_error:
                best_error = total_error
                best_bias = accel_bias.copy()
                best_sensitivity = accel_sensitivity.copy()
                best_accel = calib_accel.copy()
                print(
                    f"New best: bias={bias}, sensitivity={sensitivity}, error={total_error:.6f}")
                print(
                    f"  Roll error: {roll_error:.6f}, Pitch error: {pitch_error:.6f}")

    # Final best parameters
    print("\nFinal best accelerometer parameters:")
    print(f"  Bias: {best_bias}")
    print(f"  Sensitivity: {best_sensitivity} mV/(m/s²)")
    print(f"  Total orientation error: {best_error:.6f}")

    return best_accel


def plot_accel_calibration(calib_accel, timestamps):
    """
    Plots calibrated accelerometer data and compares it with Vicon ground-truth.

    Steps:
      1. Compute the acceleration magnitude from the calibrated accelerometer data.
      2. Compute roll and pitch from the accelerometer.
      3. Compute roll and pitch from the Vicon rotation matrices.
      4. Plot acceleration magnitude, roll, and pitch.
    """
    # Compute acceleration magnitude
    accel_mag = np.sqrt(np.sum(calib_accel ** 2, axis=0))

    roll_acc, pitch_acc = get_roll_pitch_accel(calib_accel)

    roll_vicon, pitch_vicon = ROLL_VICON, PITCH_VICON

    # Use the minimum number of samples between IMU and Vicon data
    min_samples = min(len(roll_acc), len(roll_vicon), len(timestamps))

    # Align timestamps
    t_common = timestamps[:min_samples]

    # Plot the acceleration magnitude
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(t_common, accel_mag[:min_samples], label="Accel Magnitude")
    plt.title("Acceleration Magnitude (should be ~9.81 m/s² at rest)")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()

    # Plot roll comparison
    plt.subplot(3, 1, 2)
    plt.plot(t_common, roll_acc[:min_samples], label="Accel-derived Roll")
    plt.plot(t_common, roll_vicon[:min_samples], label="Vicon Roll",
             linestyle="--")
    plt.title("Roll Comparison")
    plt.xlabel("Time")
    plt.ylabel("Roll (radians)")
    plt.legend()

    # Plot pitch comparison
    plt.subplot(3, 1, 3)
    plt.plot(t_common, pitch_acc[:min_samples], label="Accel-derived Pitch")
    plt.plot(t_common, pitch_vicon[:min_samples], label="Vicon Pitch",
             linestyle="--")
    plt.title("Pitch Comparison")
    plt.xlabel("Time")
    plt.ylabel("Pitch (radians)")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Step 1: Load data
accel, gyro, timestamps, vicon_rot = load_data(data_folder='hw2_p2_data',
                                               data_num=1)

ROLL_VICON, PITCH_VICON, YAW_Vicon = extract_vicon_values(vicon_rot)

print(f"Loaded {timestamps.shape[0]} IMU timestamps")
print(f"IMU Accelerometer shape: {accel.shape}")
print(f"IMU Gyroscope shape: {gyro.shape}")
print(f"Vicon Rotation shape: {vicon_rot.shape}")

calib_accel = find_best_calib_accel(accel)

# Step 3: Plot and compare accelerometer-derived values with Vicon ground
plot_accel_calibration(calib_accel, timestamps)