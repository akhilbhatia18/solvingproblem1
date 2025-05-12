import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Layer, Lambda
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from vmdpy import VMD
import matplotlib.pyplot as plt
import random
import math
import os
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class AttentionLayer(Layer):
    """
    Attention layer for GRU model
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


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


class IWOA_HyperOpt:
    """
    Improved Whale Optimization Algorithm for hyperparameter optimization
    """
    def __init__(self, num_whales=15, max_iter=30, b=1, a_min=0, a_max=2,
                 param_ranges=None):
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.b = b
        self.a_min = a_min
        self.a_max = a_max
        self.param_ranges = param_ranges
        self.convergence_curve = np.zeros(max_iter)
        
    def initialize_population(self):
        """Initialize the whale population with random hyperparameters"""
        population = []
        
        for _ in range(self.num_whales):
            # Generate random values within ranges for each hyperparameter
            whale = {}
            for param, param_range in self.param_ranges.items():
                if isinstance(param_range[0], bool) and isinstance(param_range[1], bool):
                    whale[param] = random.choice(param_range)
                elif isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    # Integer parameter
                    if param_range[0] < param_range[1]:
                        whale[param] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        raise ValueError(f"Invalid range for parameter '{param}': {param_range}")
                elif param in ['optimizer', 'activation']:
                    # Categorical parameter (as list of options)
                    whale[param] = random.choice(param_range)
                else:
                    # Float parameter
                    whale[param] = np.random.uniform(param_range[0], param_range[1])
                    
            population.append(whale)
            
        return population
    
    def calculate_fitness(self, hyperparams, X_train, y_train, X_val, y_val):
        """Build and evaluate model with given hyperparameters"""
        try:
            # Build model with given hyperparameters
            model = self.build_model(hyperparams, X_train.shape[1:])
            
            # Compile model
            model.compile(
                optimizer=hyperparams['optimizer'],
                loss='mse',
                metrics=['mae']
            )
            
            # Create early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=20,  # Reduced for optimization speed
                batch_size=hyperparams['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Get validation loss (fitness)
            val_loss = min(history.history['val_loss'])
            
            # Clear model and free memory
            K.clear_session()
            
            return val_loss
            
        except Exception as e:
            print(f"Error in model training: {e}")
            # Return high penalty
            return 1e10
    
    def build_model(self, hyperparams, input_shape):
        """Build GRU model with attention based on hyperparameters"""
        # Input layer
        inputs = Input(shape=input_shape)
        
        # GRU layers
        x = inputs
        for i in range(hyperparams['num_gru_layers']):
            return_sequences = (i < hyperparams['num_gru_layers'] - 1) or hyperparams['use_attention']
            x = GRU(
                units=hyperparams['gru_units'],
                activation=hyperparams['activation'],
                return_sequences=return_sequences
            )(x)
            
            # Add dropout after each GRU layer
            x = Dropout(hyperparams['dropout_rate'])(x)
            
        # Attention layer (if enabled)
        if hyperparams['use_attention']:
            x = AttentionLayer()(x)
            
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs, outputs)
        
        return model
    
    def levy_flight(self):
        """Generate Levy flight for improved exploration"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        # Create new whale with levy flight
        new_whale = {}
        for param, param_range in self.param_ranges.items():
            if isinstance(param_range[0], bool) and isinstance(param_range[1], bool):
                # Boolean parameter
                new_whale[param] = random.choice(param_range)
            elif isinstance(param_range[0], int) and isinstance(param_range[1], int):
                # Integer parameter
                u = np.random.normal(0, sigma)
                v = np.random.normal(0, 1)
                step = u / (np.abs(v) ** (1 / beta))
                
                # Scale step to parameter range
                step_scale = (param_range[1] - param_range[0]) * 0.1
                step = int(step * step_scale)
                
                # Generate new value
                new_val = np.random.randint(param_range[0], param_range[1] + 1) + step
                new_val = max(param_range[0], min(param_range[1], new_val))
                new_whale[param] = new_val
                
            elif param in ['optimizer', 'activation']:
                # Categorical parameter (as list of options)
                new_whale[param] = random.choice(param_range)
                
            else:
                # Float parameter
                u = np.random.normal(0, sigma)
                v = np.random.normal(0, 1)
                step = u / (np.abs(v) ** (1 / beta))
                
                # Scale step to parameter range
                step_scale = (param_range[1] - param_range[0]) * 0.1
                step = step * step_scale
                
                # Generate new value
                new_val = np.random.uniform(param_range[0], param_range[1]) + step
                new_val = max(param_range[0], min(param_range[1], new_val))
                new_whale[param] = new_val
                
        return new_whale
    
    def update_position(self, whale, best_whale, a, A, C, p, l):
        """Update whale position based on IWOA rules"""
        new_whale = {}
        
        for param, value in whale.items():
            param_range = self.param_ranges[param]
            
            if p < 0.5:
                # Exploitation or exploration
                if abs(A) < 1:
                    # Exploitation
                    if isinstance(value, (int, float)) and not param in ['optimizer', 'activation']:
                        # Numeric parameter
                        D = abs(C * best_whale[param] - value)
                        new_value = best_whale[param] - A * D
                        
                        # Bound check
                        if isinstance(value, int):
                            new_value = int(round(new_value))
                            new_value = max(param_range[0], min(param_range[1], new_value))
                        else:
                            new_value = max(param_range[0], min(param_range[1], new_value))
                            
                        new_whale[param] = new_value
                    else:
                        # Categorical parameter - 50% chance to adopt best position
                        new_whale[param] = best_whale[param] if random.random() < 0.5 else value
                else:
                    # Exploration
                    if isinstance(value, (int, float)) and not param in ['optimizer', 'activation', 'use_attention']:
                        # Generate new random position for numeric parameter
                        if isinstance(value, int):
                            new_whale[param] = np.random.randint(param_range[0], param_range[1] + 1)
                        else:
                            new_whale[param] = np.random.uniform(param_range[0], param_range[1])
                    else:
                        # Categorical parameter - random selection from options
                        new_whale[param] = random.choice(param_range)
            else:
                # Spiral update
                if isinstance(value, (int, float)) and not param in ['optimizer', 'activation']:
                    # Numeric parameter
                    D = abs(best_whale[param] - value)
                    spiral_factor = D * np.exp(self.b * l) * np.cos(2 * np.pi * l)
                    new_value = best_whale[param] - spiral_factor
                    
                    # Bound check
                    if isinstance(value, int):
                        new_value = int(round(new_value))
                        new_value = max(param_range[0], min(param_range[1], new_value))
                    else:
                        new_value = max(param_range[0], min(param_range[1], new_value))
                        
                    new_whale[param] = new_value
                else:
                    # Categorical parameter - 30% chance to adopt best position
                    new_whale[param] = best_whale[param] if random.random() < 0.3 else value
                    
        return new_whale
    
    def optimize(self, X_train, y_train, X_val, y_val):
        """Run optimization to find best hyperparameters"""
        # Initialize population
        population = self.initialize_population()
        
        # Calculate initial fitness for each whale
        fitness = []
        for whale in population:
            fit = self.calculate_fitness(whale, X_train, y_train, X_val, y_val)
            fitness.append(fit)
            
        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_whale = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Main optimization loop
        for t in range(self.max_iter):
            print(f"IWOA Iteration {t+1}/{self.max_iter}")
            
            # Parameter a decreases linearly from a_max to a_min
            a = self.a_max - t * ((self.a_max - self.a_min) / self.max_iter)
            
            # Update each whale's position
            for i in range(self.num_whales):
                # Random parameters
                r1 = random.random()
                r2 = random.random()
                p = random.random()
                
                # Calculate A and C
                A = 2 * a * r1 - a
                C = 2 * r2
                
                # Calculate l for spiral update
                l = (a * 2) * random.random() - a
                
                # Update position
                new_whale = self.update_position(population[i], best_whale, a, A, C, p, l)
                
                # Small chance for Levy flight to enhance exploration
                if random.random() < 0.1:
                    levy_whale = self.levy_flight()
                    
                    # Calculate fitness for Levy whale
                    levy_fitness = self.calculate_fitness(levy_whale, X_train, y_train, X_val, y_val)
                    
                    # Replace with Levy whale if better
                    if levy_fitness < fitness[i]:
                        new_whale = levy_whale
                        fitness[i] = levy_fitness
                
                # Calculate fitness for new position
                new_fitness = self.calculate_fitness(new_whale, X_train, y_train, X_val, y_val)
                
                # Update position if fitness improved
                if new_fitness < fitness[i]:
                    population[i] = new_whale
                    fitness[i] = new_fitness
                    
                    # Update global best if needed
                    if new_fitness < best_fitness:
                        best_whale = new_whale.copy()
                        best_fitness = new_fitness
                        print(f"New best fitness: {best_fitness}")
                        print(f"New best hyperparameters: {best_whale}")
            
            # Record convergence
            self.convergence_curve[t] = best_fitness
            
            # Early stopping if fitness is low enough
            if best_fitness < 0.01:
                print(f"Early stopping at iteration {t+1} with fitness {best_fitness}")
                break
                
        return best_whale, self.convergence_curve


def load_telecom_data(file_path):
    """
    Load and preprocess Shanghai Telecom dataset with location clustering
    """
    # Load the dataset
    df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])
    
    # Calculate session duration
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60  # in minutes
    
    # Extract coordinates for clustering
    coords = df[["latitude", "longitude"]].values
    
    # Apply KMeans clustering to identify edge nodes
    n_clusters = 50  # Number of edge nodes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    df["edge_node"] = kmeans.labels_
    
    # Save edge node info
    clusters_df = df[["edge_node", "latitude", "longitude"]].drop_duplicates()
    clusters_df.to_csv("clusters.csv", index=False)
    print(f"Created {n_clusters} edge node clusters")
    
    # Aggregate data by datetime and edge node
    df["hour"] = df["start_time"].dt.hour
    df["date"] = df["start_time"].dt.date
    
    # Group by date, hour, and edge node
    grouped_data = df.groupby(["date", "hour", "edge_node"]).agg({
        "user_id": "count",        # Count of active users
        "duration": "sum",         # Total session duration
        "latitude": "mean",        # Average latitude for the group
        "longitude": "mean"        # Average longitude for the group
    }).reset_index()
    
    # Rename columns for clarity
    grouped_data.rename(columns={"user_id": "active_users"}, inplace=True)
    
    return df, grouped_data, clusters_df


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
        best_params, _ = iwoa.optimize(scaled_data)
        
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
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
        
    np.savez("processed_data/location_based_imfs.npz", 
             imfs=edge_node_imfs, 
             params=edge_node_params)
    
    return edge_node_imfs, edge_node_params


def create_sequences(data, n_steps_in, n_steps_out=1):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in), :])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])
    return np.array(X), np.array(y)


def prepare_forecast_data(imfs, time_series_data, forecast_column='active_users'):
    """Prepare data for forecasting model using IMFs"""
    # Scale the original data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(time_series_data[[forecast_column]].values)
    
    # Create features for each IMF
    n_imfs = len(imfs)
    
    # Create a dataset with original data and imfs as features
    min_length = min(len(scaled_data), imfs.shape[1])
    X_data = np.zeros((min_length, n_imfs + 1))
    X_data[:, 0] = scaled_data.flatten()[:min_length]
    
    # Add IMFs as features
    for i in range(n_imfs):
        imf_data = imfs[i, :min_length]
        X_data[:, i+1] = imf_data
    
    return X_data, scaler


def build_optimized_model(best_hyperparams, input_shape):
    """Build model with optimized hyperparameters"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # GRU layers
    x = inputs
    for i in range(best_hyperparams['num_gru_layers']):
        return_sequences = (i < best_hyperparams['num_gru_layers'] - 1) or best_hyperparams['use_attention']
        x = GRU(
            units=best_hyperparams['gru_units'],
            activation=best_hyperparams['activation'],
            return_sequences=return_sequences
        )(x)
        
        # Add dropout after each GRU layer
        x = Dropout(best_hyperparams['dropout_rate'])(x)
        
    # Attention layer (if enabled)
    if best_hyperparams['use_attention']:
        x = AttentionLayer()(x)
        
    # Output layer
    outputs = Dense(1)(x)
    
    # Create model
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=best_hyperparams['optimizer'],
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_location_based_models(grouped_data, edge_node_imfs, n_steps_in=24, n_steps_out=6):
    """Train forecasting models for each edge node location"""
    # Dictionary to store models and results for each edge node
    location_models = {}
    location_results = {}
    
    # Define hyperparameter ranges for IWOA optimization
    param_ranges = {
        'num_gru_layers': [2, 4],
        'gru_units': [32, 256],
        'dropout_rate': [0.1, 0.5],
        'batch_size': [8, 64],
        'activation': ['tanh', 'relu', 'sigmoid'],
        'optimizer': ['adam', 'rmsprop'],
        'use_attention': [True, False]
    }
    
    # Create directory for model storage
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Process each edge node
    for edge_node, imfs in edge_node_imfs.items():
        print(f"\nTraining model for edge node {edge_node}")
        
        # Filter data for this edge node
        edge_data = grouped_data[grouped_data["edge_node"] == edge_node].sort_values(["date", "hour"])
        
        # Prepare forecast data
        X_data, scaler = prepare_forecast_data(imfs, edge_data)
        
        # Create sequences
        X, y = create_sequences(X_data, n_steps_in, n_steps_out)
        
        # Skip if not enough sequences
        if len(X) < 50:  # Arbitrary minimum for meaningful training
            print(f"Skipping edge node {edge_node}: insufficient sequences ({len(X)})")
            continue
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val