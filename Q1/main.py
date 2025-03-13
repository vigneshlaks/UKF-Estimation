import numpy as np
import math
import matplotlib.pyplot as plt

def get_observations(num_steps: int):
    """
    Simulates the nonlinear dynamical system.
    Samples x₀ from N(1, 2), then for each step:
      - Samples measurement noise νₖ ~ N(0, 1/2) to compute yₖ = xₖ² + 1 + νₖ.
      - Samples process noise ϵₖ ~ N(0, 1) to update xₖ₊₁ = -1 * xₖ + ϵₖ.
    Returns the list of observations (yₖ) and states (xₖ).
    """
    # Initial state
    x_0 = np.random.normal(1, math.sqrt(2))
    states = [x_0]
    observations = []

    # True parameter value
    true_a = -1

    for i in range(num_steps):
        # Calculate observation: y_k = x_k^2 + 1 + v_k
        measurement_noise = np.random.normal(0, math.sqrt(1 / 2))
        observation = states[-1] ** 2 + 1 + measurement_noise
        observations.append(observation)

        # Update state: x_{k+1} = a*x_k + epsilon_k
        process_noise = np.random.normal(0, 1)
        new_state = true_a * states[-1] + process_noise
        states.append(new_state)

    return observations, states


def ekf_parameter_estimation(observations, initial_state_mean,
                             initial_state_var,
                             initial_param_mean, initial_param_var,
                             process_noise_var, measurement_noise_var):
    """
    Extended Kalman Filter for joint state and parameter estimation
    """
    # Number of iterations (observations)
    iterations = len(observations)

    # Initialize augmented state [x, a] and covariance
    # Start with a negative value for a to help convergence
    z_mean = np.array([initial_state_mean, initial_param_mean])
    P = np.diag([initial_state_var, initial_param_var])

    # Initialize results storage
    param_means = np.zeros(iterations)
    param_vars = np.zeros(iterations)
    state_means = np.zeros(iterations)
    state_vars = np.zeros(iterations)

    # Process noise covariance (affects only x, not a)
    Q = np.diag([process_noise_var,
                 0.001])  # Add small noise to parameter to prevent filter convergence too quickly

    # Measurement noise
    R = measurement_noise_var

    # Store initial state as first estimate
    param_means[0] = z_mean[1]
    param_vars[0] = P[1, 1]
    state_means[0] = z_mean[0]
    state_vars[0] = P[0, 0]

    for k in range(1, iterations):
        # --------- Prediction step ---------
        x_prev = z_mean[0]
        a_prev = z_mean[1]

        # Predict next state
        x_pred = a_prev * x_prev
        a_pred = a_prev  # Parameter is constant
        z_pred = np.array([x_pred, a_pred])

        # Jacobian of state transition function
        F = np.array([
            [a_prev, x_prev],  # ∂f₁/∂x, ∂f₁/∂a
            [0, 1]  # ∂f₂/∂x, ∂f₂/∂a
        ])

        # Predict covariance
        P_pred = F @ P @ F.T + Q

        # --------- Update step ---------
        # Get observation
        y_k = observations[k]

        # Predicted measurement
        h_pred = x_pred ** 2 + 1

        # Jacobian of measurement function
        H = np.array([2 * x_pred, 0])  # [∂h/∂x, ∂h/∂a]

        # Innovation (measurement residual)
        innovation = y_k - h_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T / S

        # Update state and covariance
        z_mean = z_pred + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        # Store results
        param_means[k] = z_mean[1]
        param_vars[k] = P[1, 1]
        state_means[k] = z_mean[0]
        state_vars[k] = P[0, 0]

        # Debug information every few steps
        if k % 10 == 0:
            print(
                f"Step {k}: a={z_mean[1]:.4f}, x={z_mean[0]:.4f}, innovation={innovation:.4f}")

    return param_means, param_vars, state_means, state_vars


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of observations
    num_steps = 100

    # Generate observations
    observations, true_states = get_observations(num_steps)
    print(f"Generated {len(observations)} observations")

    # Try with a different initial guess
    initial_a_mean = -0.5  # Start with an informed guess closer to -1
    initial_a_var = 1.0  # Moderate uncertainty

    mu_a, sigma_a, mu_x, sigma_x = ekf_parameter_estimation(
        observations=observations,
        initial_state_mean=1,  # x_0 ~ N(1, 2)
        initial_state_var=2,  # x_0 ~ N(1, 2)
        initial_param_mean=initial_a_mean,
        initial_param_var=initial_a_var,
        process_noise_var=1,  # ε_k ~ N(0, 1)
        measurement_noise_var=0.5  # ν_k ~ N(0, 1/2)
    )

    # Plot the results
    time_steps = np.arange(len(mu_a))

    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, mu_a, marker='.', markersize=4,
             label=r'Estimated $\mu_k$ (a)')
    plt.fill_between(time_steps, mu_a - np.sqrt(sigma_a),
                     mu_a + np.sqrt(sigma_a),
                     color='gray', alpha=0.4, label=r'$\mu_k \pm \sigma_k$')
    plt.axhline(-1, color='red', linestyle='--', label=r'True $a = -1$')
    plt.xlabel('Time step k', fontsize=12)
    plt.ylabel('Parameter estimate a', fontsize=12)
    plt.title('EKF Parameter Estimation for a (True a = -1)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(-2.5, 0.5)  # Focus on the relevant range
    plt.tight_layout()
    plt.savefig('parameter_estimation.png')
    plt.show()

    # Also plot state estimation
    plt.figure(figsize=(12, 7))
    plt.plot(time_steps, mu_x, marker='.', markersize=4,
             label=r'Estimated state $x_k$')
    plt.plot(range(len(true_states) - 1), true_states[:-1], 'r--',
             label='True state')
    plt.xlabel('Time step k', fontsize=12)
    plt.ylabel('State x', fontsize=12)
    plt.title('EKF State Estimation', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('state_estimation.png')
    plt.show()

    # Print final estimate
    print(
        f"Final parameter estimate: {mu_a[-1]:.4f} ± {np.sqrt(sigma_a[-1]):.4f}")
    print(f"True parameter value: -1.0000")