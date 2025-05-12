import numpy as np
import pandas as pd
from scipy.optimize import minimize
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import math
from sklearn.cluster import KMeans
import pso_vmd_updated as pso
import time

class IWOA:
    """
    Improved Whale Optimization Algorithm for VMD parameter optimization
    """
    def __init__(self, num_whales=30, max_iter=50, b=1, a_min=0, a_max=2, 
                 lower_bounds=None, upper_bounds=None):
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.b = b
        self.a_min = a_min
        self.a_max = a_max
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.convergence_curve = np.zeros(max_iter)
        
    def initialize_population(self, dimension):
        """Initialize the whale population"""
        population = np.zeros((self.num_whales, dimension))
        for i in range(dimension):
            population[:, i] = np.random.uniform(
                self.lower_bounds[i], self.upper_bounds[i], self.num_whales)
        return population
    
    def levy_flight(self, dimension):
        """Generate Levy flight for improved exploration"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        u = np.random.normal(0, sigma, dimension)
        v = np.random.normal(0, 1, dimension)
        step = u / (np.abs(v) ** (1 / beta))
        
        # Scale step to boundaries
        step_scale = np.min(np.array(self.upper_bounds) - np.array(self.lower_bounds)) * 0.1
        step = step * step_scale
        
        # Start position (random in bounds)
        pos = np.array([np.random.uniform(self.lower_bounds[i], self.upper_bounds[i]) 
                        for i in range(dimension)])
        
        return pos
    
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
        """Optimize VMD parameters using IWOA"""
        # Parameters to optimize: K, alpha, tau, DC
        dimension = 4
        fitness = np.zeros(self.num_whales)
        
        # Initialize positions
        positions = self.initialize_population(dimension)
        
        # Initialize best position
        fitness = np.array([self.calculate_fitness(pos, data) for pos in positions])
        best_idx = np.argmin(fitness)
        best_pos = positions[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Main optimization loop
        for t in range(self.max_iter):
            a = self.a_max - t * ((self.a_max - self.a_min) / self.max_iter)
            
            # Update each whale's position
            for i in range(self.num_whales):
                # Calculate r1, r2, p
                r1 = random.random()
                r2 = random.random()
                p = random.random()
                
                # Calculate A and C
                A = 2 * a * r1 - a
                C = 2 * r2
                
                # Calculate l for spiral update
                l = (a * 2) * random.random() - a
                
                # Decide whether to perform exploitation or exploration
                if p < 0.5:
                    # Perform exploitation (search prey)
                    if abs(A) < 1:
                        # Update position using encircling prey mechanism
                        D = abs(C * best_pos - positions[i])
                        new_pos = best_pos - A * D
                    else:
                        # Exploration (search for prey)
                        random_idx = np.random.randint(0, self.num_whales)
                        random_whale = positions[random_idx]
                        D = abs(C * random_whale - positions[i])
                        new_pos = random_whale - A * D
                else:
                    # Perform spiral update
                    D = abs(best_pos - positions[i])
                    spiral_pos = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_pos
                    
                    # Improved: Levy flight to enhance exploration
                    levy = self.levy_flight(dimension)
                    
                    # Adaptive weight based on fitness
                    weight = 0.5 + 0.5 * np.exp(-fitness[i] / best_fitness)
                    new_pos = weight * spiral_pos + (1 - weight) * levy
                
                # Clamp to boundaries
                for j in range(dimension):
                    new_pos[j] = np.clip(new_pos[j], self.lower_bounds[j], self.upper_bounds[j])
                
                # Calculate fitness of new position
                new_fitness = self.calculate_fitness(new_pos, data)
                
                # Update position if fitness improved
                if new_fitness < fitness[i]:
                    positions[i] = new_pos
                    fitness[i] = new_fitness
                    
                    # Update global best if needed
                    if new_fitness < best_fitness:
                        best_pos = new_pos.copy()
                        best_fitness = new_fitness
            
            # Record convergence
            self.convergence_curve[t] = best_fitness
            
        # Round K to nearest integer
        best_pos[0] = round(best_pos[0])
        
        # Return the best parameters: K, alpha, tau, DC
        optimized_params = {
            'K': int(best_pos[0]),
            'alpha': best_pos[1],
            'tau': best_pos[2],
            'DC': best_pos[3] > 0.5
        }
        
        return optimized_params, self.convergence_curve


def load_shanghai_telecom_data(file_path):
    """
    Load and preprocess Shanghai Telecom dataset
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Parse date and time columns
    data['datetime'] = pd.to_datetime(data['start_time'], dayfirst=True, format='mixed')

    # Calculate session duration
    data['end_time'] = pd.to_datetime(data['end_time'], dayfirst=True, format='mixed')
    data['duration'] = (data['end_time'] - data['datetime']).dt.total_seconds() / 60  # in minutes
    
    # Extract coordinates for clustering
    coords = data[["latitude", "longitude"]].values
    # Handle NaN values in coordinates
    if np.isnan(coords).any():
        print("NaN values detected in coordinates. Handling missing values...")
        # Option 1: Drop rows with NaN values
        data = data.dropna(subset=["latitude", "longitude"])
        coords = data[["latitude", "longitude"]].values
    
    # Apply KMeans clustering to identify edge nodes
    n_clusters = 2  # Number of edge nodes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    data["edge_node"] = kmeans.labels_

    # Save edge node info
    clusters_df = data[["edge_node", "latitude", "longitude"]].drop_duplicates()
    clusters_df.to_csv("clusters.csv", index=False)
    print(f"Created {n_clusters} edge node clusters")

    # Aggregate data by datetime and edge node
    data["hour"] = data["datetime"].dt.hour
    data["date"] = data["datetime"].dt.date
    
    # Group by date, hour, and edge node - Fixed the typo from "hours" to "hour"
    grouped_data = data.groupby(["date", "hour", "edge_node"]).agg({
        "user_id": "count",        # Count of active users
        "duration": "sum",         # Total session duration
        "latitude": "mean",        # Average latitude for the group
        "longitude": "mean"        # Average longitude for the group
    }).reset_index()
    
    # Rename columns for clarity
    grouped_data.rename(columns={"user_id": "active_users"}, inplace=True)
    
    return data, grouped_data, clusters_df

def optimize_vmd_for_edge_node(edge_node_data):
    """
    Apply VMD optimization for a specific edge node's time series
    """
    # Extract the time series
    time_series = edge_node_data["active_users"].values
    
    # Skip if not enough data points
    if len(time_series) < 10:
        return None, None
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    
    # Initialize IWOA with parameter bounds
    iwoa = IWOA(
        num_whales=10, 
        max_iter=15,
        lower_bounds=[3, 500, 0, 0],  # K, alpha, tau, DC
        upper_bounds=[8, 5000, 1, 1]
    )
    
    # Run optimization
    try:
        best_params, convergence = iwoa.optimize(scaled_data)
        
        # Run VMD with optimized parameters
        K = best_params['K']
        alpha = best_params['alpha']
        tau = best_params['tau']
        DC = best_params['DC']
        
        u, _, _ = VMD(scaled_data, alpha, tau, K, DC, 1, 1e-6)
        
        # Return IMFs and parameters
        return u, best_params
    except Exception as e:
        print(f"Error in VMD optimization: {e}")
        return None, None
    
def calculate_metrics(original, reconstructed):
    """
    Calculate MAPE, SMAPE, and MAE for the reconstructed signal.
    """
    epsilon = 1e-10  # To avoid division by zero
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((original - reconstructed) / (original + epsilon))) * 100
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = 100 * np.mean(2.0 * np.abs(original - reconstructed) / (np.abs(original) + np.abs(reconstructed) + epsilon))

    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(original - reconstructed))

    mse = np.mean((original - reconstructed) ** 2)
    
    return mape, smape, mae, mse

def process_with_vmd_iwoa(time_series):
    """
    Process time series data with VMD using IWOA-optimized parameters
    """
    # Skip if not enough data points
    if len(time_series) < 10:
        return None, None, None
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    
    # Initialize IWOA with parameter bounds
    # K: 3-8, alpha: 500-5000, tau: 0-1, DC: 0-1 (binary)
    iwoa = IWOA(
        num_whales=10, 
        max_iter=15,
        lower_bounds=[3, 500, 0, 0],
        upper_bounds=[8, 5000, 1, 1]
    )
    
    # Run optimization
    best_params, convergence = iwoa.optimize(scaled_data)
    print(f"Optimized VMD parameters: {best_params}")
    
    # Run VMD with optimized parameters
    K = best_params['K']
    alpha = best_params['alpha']
    tau = best_params['tau']
    DC = best_params['DC']
    
    u, u_hat, omega = VMD(scaled_data, alpha, tau, K, DC, 1, 1e-6)
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(convergence)
    plt.title('IWOA Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Error)')
    plt.savefig('iwoa_convergence.png')
    
    # Plot IMFs
    plt.figure(figsize=(12, 8))
    for i in range(K):
        plt.subplot(K, 1, i+1)
        plt.plot(u[i, :])
        plt.ylabel(f'IMF {i+1}')
    plt.tight_layout()
    plt.savefig('vmd_imfs.png')
    
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
    
    return u, reconstructed_original, best_params, (mape, smape, mae, mse)


def process_location_based_imfs(grouped_data):
    """
    Process IMFs for each edge node location
    """
    # Dictionary to store IMFs for each edge node
    edge_node_imfs = {}
    edge_node_params = {}
    
    # Get unique edge nodes
    edge_nodes = grouped_data["edge_node"].unique()
    
    for edge_node in edge_nodes:
        print(f"Processing edge node {edge_node}")
        
        # Filter data for this edge node
        edge_data = grouped_data[grouped_data["edge_node"] == edge_node].sort_values(["date", "hour"])
        
        # Skip if not enough data
        if len(edge_data) < 24:  # Need at least a day of data
            print(f"Skipping edge node {edge_node}: insufficient data ({len(edge_data)} points)")
            continue
            
        # Optimize VMD and extract IMFs
        imfs, params = optimize_vmd_for_edge_node(edge_data)
        
        if imfs is not None:
            edge_node_imfs[edge_node] = imfs
            edge_node_params[edge_node] = params
    
    # Save processed IMFs
    import os
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
        
    np.savez("processed_data/location_based_imfs.npz", 
             imfs=edge_node_imfs, 
             params=edge_node_params)
    
    return edge_node_imfs, edge_node_params


def main(file_path):
    """
    Main function to execute the preprocessing pipeline
    """
    # Load and preprocess data
    raw_data, hourly_data, clusters_df = load_shanghai_telecom_data(file_path)
    
    # Process IMFs for each edge node location
    edge_node_imfs, edge_node_params = process_location_based_imfs(hourly_data)
    method = "PSO"  # or "PSO"
    # Extract the active users time series (global analysis)
    active_users = hourly_data['active_users'].values

    start_time = time.time()
    if method == "IWOA":
        imfs, reconstructed, vmd_params,metrics = process_with_vmd_iwoa(active_users)
    elif method == "PSO":
        imfs, reconstructed, vmd_params,metrics = pso.optimize_vmd_with_pso(active_users)
    else:
        raise ValueError("Invalid method. Choose 'IWOA' or 'PSO'.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print metrics
    mape, smape, mae, mse = metrics
    print(f"Final Metrics - MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%, MAE: {mae:.4f}")
    print(f"Running Time: {elapsed_time:.2f} seconds")
    
    # Process with VMD and IWOA (global analysis)
    #imfs, reconstructed, vmd_params = process_with_vmd_iwoa(active_users)
    
    # Save IMFs to file
    np.savez('processed_imfs.npz', imfs=imfs, params=vmd_params)
    
    # Plot original vs reconstructed
    plt.figure(figsize=(12, 6))
    plt.plot(active_users, label='Original')
    if reconstructed is not None:
        plt.plot(reconstructed, label='Reconstructed')
    plt.legend()
    plt.title('Original vs Reconstructed Signal')
    plt.savefig('reconstruction.png')
    
    print(f"Processed {len(edge_node_imfs)} edge nodes with VMD")
    return edge_node_imfs, edge_node_params


if __name__ == "__main__":
    # Example usage
    file_path = "data/DataSet/telecom_dataset_output.csv"
    edge_node_imfs, params = main(file_path)

    print(f"Processing complete with parameters for {len(edge_node_imfs)} edge nodes")