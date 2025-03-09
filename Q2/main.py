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

def calibrate_sensors(accel, gyro):
    """
    Hardcoded calibration for accelerometer and gyroscope.

    The conversion formula is:
        value = (raw - beta) * (3300 mV) / (1023 * alpha)

    The accelerometer is then converted to m/s².
    The gyroscope is converted to rad/sec.

    Returns:
        calib_accel (ndarray): Accelerometer values in m/s².
        calib_gyro (ndarray): Gyroscope values in rad/sec.
    """
    # Hardcoded biases and sensitivities
    accel_bias = np.array([500, 500, 500])
    accel_sensitivity = np.array([30, 30, 30])  # mV/(m/s²), within provided range

    gyro_bias = np.array([350, 350, 350])
    gyro_sensitivity = np.array([200, 200, 200])  # mV/(rad/sec)

    adc_to_mV = 3300 / 1023

    # Calibrate accelerometer:
    calib_accel = (accel - accel_bias[:, None]) * (
                adc_to_mV / accel_sensitivity[:, None]) * 9.81

    # Flip due to device design or wtv
    calib_accel[0, :] = -calib_accel[0, :]
    calib_accel[1, :] = -calib_accel[1, :]

    # Do same for gyro
    calib_gyro = (gyro - gyro_bias[:, None]) * (
                adc_to_mV / gyro_sensitivity[:, None])

    return calib_accel, calib_gyro

def plot_accel_calibration(calib_accel, vicon_rot, timestamps):
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

    # Compute roll and pitch from calibrated accelerometer
    ax = calib_accel[0, :]
    ay = calib_accel[1, :]
    az = calib_accel[2, :]

    eps = 1e-6  # Small value to avoid division by zero
    roll_acc = np.arctan2(ay, az + eps)  # in radians
    pitch_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + eps)

    # Get roll and pitch from vicon
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

print(f"Loaded {timestamps.shape[0]} IMU timestamps")
print(f"IMU Accelerometer shape: {accel.shape}")
print(f"IMU Gyroscope shape: {gyro.shape}")
print(f"Vicon Rotation shape: {vicon_rot.shape}")

# Step 2: Calibrate sensors (convert raw ADC values to physical units)
calib_accel, calib_gyro = calibrate_sensors(accel, gyro)

# Step 3: Plot and compare accelerometer-derived values with Vicon ground truth.
plot_accel_calibration(calib_accel, vicon_rot, timestamps)