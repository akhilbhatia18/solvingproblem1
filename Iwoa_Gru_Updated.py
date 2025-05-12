import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Layer, Lambda
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from vmdpy import VMD
import matplotlib.pyplot as plt
import random
import math
import os
import json
import datetime
import Iwoa_Vmd_Updated as vmd

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
                    whale[param] = random.choice([True, False])
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
                new_whale[param] = random.choice([True, False])
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
                        if param in ['optimizer', 'activation']:
                            new_whale[param] = random.choice(param_range)
                        else:
                            new_whale[param] = random.choice([True, False])
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


def create_sequences(data, n_steps_in, n_steps_out=1):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in), :])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])
    return np.array(X), np.array(y)


def load_processed_imfs(imf_file='processed_data/location_based_imfs.npz'):
    """Load processed IMFs from file with location info"""
    data = np.load(imf_file, allow_pickle=True)
    edge_node_imfs = data['imfs'].item()  # Dictionary mapping edge_node to IMFs
    edge_node_params = data.get('params', None)
    
    if edge_node_params is not None and isinstance(edge_node_params, np.ndarray) and edge_node_params.size == 1:
        edge_node_params = edge_node_params.item()
    
    return edge_node_imfs, edge_node_params


def prepare_forecast_data(imfs, edge_data, forecast_column='active_users'):
    """Prepare data for forecasting model using IMFs"""
    # Scale the original data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(edge_data[[forecast_column]].values)
    
    # Create features for each IMF
    n_imfs = imfs.shape[0]
    
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

def calculate_accuracy_metrics(y_true, y_pred):
    """
    Calculate accuracy metrics for time series forecasting:
    - MAPE (Mean Absolute Percentage Error)
    - SMAPE (Symmetric Mean Absolute Percentage Error)
    - R² (Coefficient of Determination)
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - MAPE w/epsilon (Modified MAPE to handle zero values)
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing all calculated metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Handle potential zero division
    epsilon = 1e-10
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Mean Absolute Percentage Error (with zero handling)
    # Using epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100
    
    # R² (Coefficient of Determination)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / (ss_total + epsilon))
    
    # Modified MAPE (with epsilon)
    mape_with_epsilon = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Create metrics dictionary
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'R2': r2,
        'MAPE_with_epsilon': mape_with_epsilon
    }
    
    return metrics


def train_location_based_models(grouped_data, edge_node_imfs, n_steps_in=24, n_steps_out=6):
    """Train forecasting models for each edge node location"""
    # Dictionary to store models and results for each edge node
    location_models = {}
    location_results = {}
    
    # Define hyperparameter ranges for IWOA optimization
    param_ranges = {
        'num_gru_layers': [2, 4],
        'gru_units': [32, 256], # Run for 32 and 256 units
        'dropout_rate': [0.1, 0.5], --> Default
        'batch_size': [512, 512], --> default
        'activation': ['tanh', 'relu'], --> Default
        'optimizer': ['adam'], --> Default
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
        
        # Skip if not enough data
        if len(edge_data) < 50:  # Arbitrary minimum for meaningful training
            print(f"Skipping edge node {edge_node}: insufficient data ({len(edge_data)})")
            continue
            
        # Prepare forecast data
        X_data, scaler = prepare_forecast_data(imfs, edge_data)
        
        # Create sequences
        X, y = create_sequences(X_data, n_steps_in, n_steps_out)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
        
        # Run IWOA for hyperparameter optimization
        iwoa = IWOA_HyperOpt(
            num_whales=10,
            max_iter=15,
            param_ranges=param_ranges
        )
        
        # Get best hyperparameters
        best_hyperparams, convergence = iwoa.optimize(X_train, y_train, X_val, y_val)
        
        # Build model with optimized hyperparameters
        model = build_optimized_model(best_hyperparams, X_train.shape[1:])
        
        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f'models/edge_node_{edge_node}_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best_hyperparams['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Evaluate model on test set
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"test_loss shape: {test_loss}, test_mae shape: {test_mae}")

            # Get predictions for test set
        y_pred = model.predict(X_test)
    
    # Ensure y_pred matches the length of y_test
        y_pred = y_pred[:len(y_test)]
    
    # Create dummy array with same shape as input to scaler for inverse transformation
        dummy = np.zeros((len(y_test.flatten()), X_data.shape[1]))
        dummy[:, 0] = y_test.flatten()
        y_test_inv = scaler.inverse_transform(dummy)[:, 0]
    
        dummy = np.zeros((len(y_pred.flatten()), X_data.shape[1]))
        dummy[:, 0] = y_pred.flatten()
        y_pred_inv = scaler.inverse_transform(dummy)[:, 0]

        # Ensure y_pred_inv matches the length of y_test_inv
        y_pred_inv = y_pred_inv[:len(y_test_inv)]
    
         # Debugging: Print shapes
        print(f"y_test_inv shape: {y_test_inv.shape}, y_pred_inv shape: {y_pred_inv.shape}")

            # Check if shapes match before calculating metrics
        if len(y_test_inv) != len(y_pred_inv):
            print(f"Shape mismatch: y_test_inv has shape {y_test_inv.shape}, "
                         f"while y_pred_inv has shape {y_pred_inv.shape}")
                # Set default values for accuracy metrics
            accuracy_metrics = {
                'MAE': None,
                'RMSE': None,
                'MAPE': None,
                'SMAPE': None,
                'R2': None,
                'MAPE_with_epsilon': None
                }
        else:
        # Calculate accuracy metrics
            accuracy_metrics = calculate_accuracy_metrics(y_test_inv, y_pred_inv)
        # Store results
        location_models[edge_node] = model
        location_results[edge_node] = {
            'best_hyperparams': best_hyperparams,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'accuracy_metrics': accuracy_metrics,
            'convergence': convergence
        }
        
            # Save model configuration
        model_config = {
        'edge_node': edge_node,
        'best_hyperparams': best_hyperparams,
        'test_loss': float(test_loss),  # Ensure float type
        'test_mae': float(test_mae),    # Ensure float type
        'accuracy_metrics': {k: (float(v) if v is not None else None) for k, v in accuracy_metrics.items()},
        'input_shape': tuple(int(dim) for dim in X_train.shape[1:]),  # Convert to tuple of ints
        'n_steps_in': int(n_steps_in),  # Ensure int type
        'n_steps_out': int(n_steps_out) # Ensure int type
        }
        
        # Create and save predictions plot
        y_pred = model.predict(X_test)
        
        # Ensure y_pred matches the length of y_test
        y_pred = y_pred[:len(y_test)]
        
        # Create dummy array with same shape as input to scaler
        dummy = np.zeros((len(y_test.flatten()), X_data.shape[1]))
        dummy[:, 0] = y_test.flatten()
        y_test_inv = scaler.inverse_transform(dummy)[:, 0]
        
        dummy = np.zeros((len(y_pred.flatten()), X_data.shape[1]))
        dummy[:, 0] = y_pred.flatten()
        y_pred_inv = scaler.inverse_transform(dummy)[:, 0]
        
        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label='Actual', marker='o')
        plt.plot(y_pred_inv, label='Predicted', marker='x')
        plt.title(f'Edge Node {edge_node} Load Forecasting: Actual vs Predicted')
        plt.xlabel('Time Step')
        plt.ylabel('Active Users')

        # Add text with metrics
        # Safely format metrics, replacing None with "N/A"
        metrics_text = (
            f"MAPE: {accuracy_metrics['MAPE']:.2f}%\n" if accuracy_metrics['MAPE'] is not None else "MAPE: N/A\n"
            f"SMAPE: {accuracy_metrics['SMAPE']:.2f}%\n" if accuracy_metrics['SMAPE'] is not None else "SMAPE: N/A\n"
            f"R²: {accuracy_metrics['R2']:.4f}" if accuracy_metrics['R2'] is not None else "R²: N/A"
        )
        plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/edge_node_{edge_node}_forecast.png')
        plt.close()

        # Save scaler for future use
        with open(f'models/edge_node_{edge_node}_scaler.pkl', 'wb') as f:
            import pickle
            pickle.dump(scaler, f)
            
        # Generate future forecasts with timestamps and location
        last_sequence = X[-1]
        forecast_df = forecast_future_load_with_location(model, last_sequence, scaler, 
                                                         edge_node, edge_data, n_steps=24)
        
        # Save forecast to CSV
        forecast_df.to_csv(f'results/edge_node_{edge_node}_forecast.csv', index=False)
        print(f"Completed training and forecasting for edge node {edge_node}")
    
    # Save summary of all models
    summary = {edge_node: {
        'test_mae': results['test_mae'],
        'hyperparams': results['best_hyperparams']
    } for edge_node, results in location_results.items()}
    
    with open('results/all_models_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    return location_models, location_results

def forecast_future_load_with_location(model, last_sequence, scaler, edge_node, edge_data, n_steps=24):
    """Forecast future load using trained model with timestamps and location"""
    future_preds = []
    current_seq = last_sequence.copy()
    
    # Get the last known timestamp from edge_data
    last_timestamp = pd.Timestamp(
        edge_data.iloc[-1]['date'].strftime('%Y-%m-%d') + ' ' + 
        f"{int(edge_data.iloc[-1]['hour']):02d}:00:00"
    )
    
    # Get location information for this edge node
    edge_latitude = edge_data['latitude'].iloc[-1]
    edge_longitude = edge_data['longitude'].iloc[-1]
    
    # Generate future timestamps
    timestamps = []
    for i in range(n_steps):
        # Predict next step
        next_pred = model.predict(current_seq.reshape(1, *current_seq.shape))
        
        # Add prediction to result
        future_preds.append(next_pred[0, 0])
        
        # Update sequence for next prediction (sliding window)
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, 0] = next_pred[0, 0]
        
        # Generate timestamp for this prediction (1 hour increments)
        next_timestamp = last_timestamp + pd.Timedelta(hours=i+1)
        timestamps.append(next_timestamp)
    
    # Create dummy array for inverse transform
    dummy = np.zeros((len(future_preds), current_seq.shape[1]))
    dummy[:, 0] = future_preds
    
    # Inverse transform predictions
    future_preds_inv = scaler.inverse_transform(dummy)[:, 0]
    
    # Create DataFrame with timestamps, location, and predictions
    forecast_df = pd.DataFrame({
        'timestamp': timestamps,
        'edge_node': edge_node,
        'latitude': edge_latitude,
        'longitude': edge_longitude,
        'predicted_active_users': future_preds_inv
    })
    
    return forecast_df


def main():
    """Main function to train location-based forecasting models"""
    # Load telecom data
    print("Loading telecom data...")
    _, grouped_data, clusters_df = vmd.load_shanghai_telecom_data('data/telecom_dataset_output.csv')
    
    # Load processed IMFs for each edge node
    print("Loading processed IMFs...")
    edge_node_imfs, edge_node_params = load_processed_imfs()
    
    # Train models for each edge node location
    print("Training location-based forecasting models...")
    location_models, location_results = train_location_based_models(
        grouped_data, edge_node_imfs, n_steps_in=24, n_steps_out=6
    )
    
    # Generate aggregated forecast for all edge nodes
    print("Generating aggregated forecast across all locations...")
    
    # Combine all forecasts into a single DataFrame
    all_forecasts = []
    for edge_node in location_models.keys():
        forecast_file = f'results/edge_node_{edge_node}_forecast.csv'
        if os.path.exists(forecast_file):
            edge_forecast = pd.read_csv(forecast_file, parse_dates=['timestamp'])
            all_forecasts.append(edge_forecast)
    
    if all_forecasts:
        # Concatenate all forecasts
        combined_forecast = pd.concat(all_forecasts)
        
        # Save combined forecast
        combined_forecast.to_csv('results/all_locations_forecast.csv', index=False)
        
        # Create visualization of forecasts across locations
        plt.figure(figsize=(14, 8))
        
        # Group by timestamp and sum predictions
        total_by_time = combined_forecast.groupby('timestamp')['predicted_active_users'].sum().reset_index()
        
        plt.plot(total_by_time['timestamp'], total_by_time['predicted_active_users'], 
                 marker='o', linestyle='-', color='blue')
        plt.title('Aggregated Network Load Forecast Across All Edge Nodes')
        plt.xlabel('Time')
        plt.ylabel('Total Active Users')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('results/aggregated_forecast.png')
        
        # Create heatmap visualization of load distribution across locations
        print("Creating geographic heatmap of forecasted load...")
        latest_forecast = combined_forecast.loc[combined_forecast['timestamp'] == 
                                              combined_forecast['timestamp'].max()]
        
        # Plot map with locations sized by predicted load
        plt.figure(figsize=(12, 10))
        plt.scatter(latest_forecast['longitude'], latest_forecast['latitude'], 
                   s=latest_forecast['predicted_active_users']/10, 
                   alpha=0.7, c='red', edgecolors='black')
        plt.title(f'Predicted Network Load Distribution at {latest_forecast["timestamp"].iloc[0]}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.savefig('results/geographic_load_distribution.png')
        
        # Print summary statistics
        print("\nForecast Summary:")
        print(f"Total edge nodes analyzed: {len(location_models)}")
        print(f"Average forecast MAE across models: {np.mean([r['test_mae'] for r in location_results.values()]):.4f}")
        print(f"Peak predicted load: {total_by_time['predicted_active_users'].max():.2f} active users")
        print(f"Time of peak load: {total_by_time.loc[total_by_time['predicted_active_users'].argmax(), 'timestamp']}")
        
        # Save summary to file
        with open('results/forecast_summary.txt', 'w') as f:
            f.write(f"Forecast generated on: {datetime.datetime.now()}\n")
            f.write(f"Total edge nodes analyzed: {len(location_models)}\n")
            f.write(f"Average forecast MAE: {np.mean([r['test_mae'] for r in location_results.values()]):.4f}\n")
            f.write(f"Peak predicted load: {total_by_time['predicted_active_users'].max():.2f} active users\n")
            f.write(f"Time of peak load: {total_by_time.loc[total_by_time['predicted_active_users'].argmax(), 'timestamp']}\n")
    
    else:
        print("No forecasts were generated. Check that models were trained successfully.")
    
    # In the main() function, update the print summary section:
# Print summary statistics
    print("\nForecast Summary:")
    print(f"Total edge nodes analyzed: {len(location_models)}")
    print(f"Average forecast MAE across models: {np.mean([r['test_mae'] for r in location_results.values()]):.4f}")

# Add these lines:
    print("\nAccuracy Metrics Across All Models:")
    avg_mape = np.mean([r['accuracy_metrics']['MAPE'] for r in location_results.values()])
    avg_smape = np.mean([r['accuracy_metrics']['SMAPE'] for r in location_results.values()])
    avg_r2 = np.mean([r['accuracy_metrics']['R2'] for r in location_results.values()])
    avg_rmse = np.mean([r['accuracy_metrics']['RMSE'] for r in location_results.values()])

    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average SMAPE: {avg_smape:.2f}%")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")

# Update the summary file writing:
    with open('results/forecast_summary.txt', 'w') as f:
        f.write(f"Forecast generated on: {datetime.datetime.now()}\n")
        f.write(f"Total edge nodes analyzed: {len(location_models)}\n")
        f.write(f"Average forecast MAE: {np.mean([r['test_mae'] for r in location_results.values()]):.4f}\n")
    
    # Add these lines:
        f.write(f"Average MAPE: {avg_mape:.2f}%\n")
        f.write(f"Average SMAPE: {avg_smape:.2f}%\n") 
        f.write(f"Average R²: {avg_r2:.4f}\n")
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")
    
        f.write(f"Peak predicted load: {total_by_time['predicted_active_users'].max():.2f} active users\n")
        f.write(f"Time of peak load: {total_by_time.loc[total_by_time['predicted_active_users'].argmax(), 'timestamp']}\n")
    print("Process completed successfully.")


if __name__ == "__main__":
    main()