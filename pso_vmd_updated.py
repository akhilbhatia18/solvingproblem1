import numpy as np
import random
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class PSO:
    """
    Particle Swarm Optimization for VMD parameter optimization
    """
    def __init__(self, num_particles=30, max_iter=50, lower_bounds=None, upper_bounds=None, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.convergence_curve = np.zeros(max_iter)

    def initialize_particles(self, dimension):
        """Initialize particle positions and velocities"""
        positions = np.zeros((self.num_particles, dimension))
        velocities = np.zeros((self.num_particles, dimension))
        for i in range(dimension):
            positions[:, i] = np.random.uniform(self.lower_bounds[i], self.upper_bounds[i], self.num_particles)
            velocities[:, i] = np.random.uniform(-1, 1, self.num_particles)
        return positions, velocities

    def calculate_fitness(self, position, data):
        """Calculate fitness based on VMD reconstruction error"""
        try:
            # Extract VMD parameters from position
            K = int(position[0])  # Number of modes
            alpha = position[1]   # Moderate bandwidth constraint
            tau = position[2]     # Noise-tolerance (no strict fidelity enforcement)
            DC = position[3] > 0.5  # True if > 0.5
            init = 1              # Initialize omegas uniformly
            tol = 1e-6           # Tolerance
            
            # Applying VMD
            _, u_hat, _ = VMD(data, alpha, tau, K, DC, init, tol)
            
            # Reconstruct the signal
            reconstructed = np.sum(u_hat, axis=0)
            
            # Calculate Mean Squared Error
            mse = np.mean((data - reconstructed) ** 2)
            
            # Add penalty for excessive modes (to prefer simpler models)
            penalty = 0.01 * K
            
            return mse + penalty
        except Exception as e:
            # Return high penalty if VMD fails
            return 1e10  # High value to avoid this solution

    def optimize(self, data):
        """Optimize VMD parameters using PSO"""
        # Parameters to optimize: K, alpha, tau, DC
        dimension = 4
        fitness = np.zeros(self.num_particles)
        
        # Initialize particles and velocities
        positions, velocities = self.initialize_particles(dimension)
        
        # Initialize personal bests and global best
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([self.calculate_fitness(pos, data) for pos in positions])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # Main optimization loop
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                r1 = random.random()
                r2 = random.random()
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                    self.c2 * r2 * (global_best_position - positions[i])
                )
                
                # Update position
                positions[i] += velocities[i]
                
                # Clamp to boundaries
                for j in range(dimension):
                    positions[i, j] = np.clip(positions[i, j], self.lower_bounds[j], self.upper_bounds[j])
                
                # Calculate fitness
                fitness[i] = self.calculate_fitness(positions[i], data)
                
                # Update personal best
                if fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_fitness[i] = fitness[i]
                    
                    # Update global best
                    if fitness[i] < global_best_fitness:
                        global_best_position = positions[i].copy()
                        global_best_fitness = fitness[i]
            
            # Record convergence
            self.convergence_curve[t] = global_best_fitness
        
        # Round K to nearest integer
        global_best_position[0] = round(global_best_position[0])
        
        # Return the best parameters: K, alpha, tau, DC
        optimized_params = {
            'K': int(global_best_position[0]),
            'alpha': global_best_position[1],
            'tau': global_best_position[2],
            'DC': global_best_position[3] > 0.5
        }
        
        return optimized_params, self.convergence_curve

def calculate_metrics(original, reconstructed):
    """
    Calculate MAPE, SMAPE, and MAE for the reconstructed signal.
    """
    epsilon = 1e-10  # To avoid division by zero
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((original - reconstructed) / (original + epsilon))) * 100
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = np.mean(2.0 * np.abs(original - reconstructed) / (np.abs(original) + np.abs(reconstructed) + epsilon)) * 100
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(original - reconstructed))
    
    mse = np.mean((original - reconstructed) ** 2)

    return mape, smape, mae,mse

def optimize_vmd_with_pso(time_series):
    """
    Process time series data with VMD using PSO-optimized parameters
    """
    # Skip if not enough data points
    if len(time_series) < 10:
        return None, None, None
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    
    # Initialize PSO with parameter bounds
    pso = PSO(
        num_particles=10, 
        max_iter=15,
        lower_bounds=[3, 500, 0, 0],  # K, alpha, tau, DC
        upper_bounds=[8, 5000, 1, 1]
    )
    
    # Run optimization
    best_params, convergence = pso.optimize(scaled_data)
    print(f"Optimized VMD parameters (PSO): {best_params}")
    
    # Run VMD with optimized parameters
    K = best_params['K']
    alpha = best_params['alpha']
    tau = best_params['tau']
    DC = best_params['DC']
    
    u, u_hat, omega = VMD(scaled_data, alpha, tau, K, DC, 1, 1e-6)
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(convergence)
    plt.title('PSO Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Error)')
    plt.savefig('pso_convergence.png')
    
    # Plot IMFs
    plt.figure(figsize=(12, 8))
    for i in range(K):
        plt.subplot(K, 1, i+1)
        plt.plot(u[i, :])
        plt.ylabel(f'IMF {i+1}')
    plt.tight_layout()
    plt.savefig('vmd_imfs_pso.png')
    
    # Reconstruct signal
    reconstructed = np.sum(u, axis=0)
    
        # Adjust the reconstructed signal to match the original signal's length
    if len(reconstructed) > len(time_series):
        reconstructed = reconstructed[:len(time_series)]  # Truncate
    elif len(reconstructed) < len(time_series):
        reconstructed = np.pad(reconstructed, (0, len(time_series) - len(reconstructed)), mode='constant')  # Pad
    
    # Inverse scaling the reconstructed signal
    reconstructed_original = scaler.inverse_transform(reconstructed.reshape(-1, 1)).flatten()
    
    mape, smape, mae, mse = calculate_metrics(time_series, reconstructed_original)
    print(f"MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%, MAE: {mae:.4f}, MSE: {mse:.2f}%")

    return u, reconstructed_original, best_params,(mape, smape, mae,mse)