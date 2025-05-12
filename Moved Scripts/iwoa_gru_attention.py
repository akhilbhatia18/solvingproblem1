import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Layer, Lambda
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import math
import os
import json
import iwoa_vmd as iwoa_vmd
import pdb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def create_sequences(data, n_steps_in, n_steps_out=1):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in), :])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])
    return np.array(X), np.array(y)


def load_processed_imfs(imf_file='processed_imfs.npz'):
    """Load processed IMFs from file"""
    data = np.load(imf_file, allow_pickle=True)
    imfs = data['imfs']
    params = data.get('params', None)
    
    if params is not None and isinstance(params, np.ndarray) and params.size == 1:
        params = params.item()
    
    return imfs, params


def prepare_forecast_data(imfs, hourly_data, forecast_column='active_users'):
    """Prepare data for forecasting model using IMFs"""
    # Scale the original data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_data[[forecast_column]])
    
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


def train_forecasting_model(X_data, scaler, n_steps_in=24, n_steps_out=6):
    """Train load forecasting model with IWOA-optimized hyperparameters"""
    # Create sequences
    X, y = create_sequences(X_data, n_steps_in, n_steps_out)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    
    # Define hyperparameter ranges
    param_ranges = {
        'num_gru_layers': [3, 5],
        'gru_units': [32, 512],
        'dropout_rate': [0.05, 0.5],
        'batch_size': [8, 64],
        'activation': ['tanh', 'relu','sigmoid'],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'use_attention': [True, False]
    }
    
    # Run IWOA for hyperparameter optimization
    iwoa = IWOA_HyperOpt(
        num_whales=10,
        max_iter=15,
        param_ranges=param_ranges
    )
    
    best_hyperparams, convergence = iwoa.optimize(X_train, y_train, X_val, y_val)
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(convergence)
    plt.title('IWOA Hyperparameter Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.savefig('iwoa_hyperopt_convergence.png')
    
    # Save best hyperparameters
    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    
    # Build model with optimized hyperparameters
    model = build_optimized_model(best_hyperparams, X_train.shape[1:])
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_forecast_model.h5',
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
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_training_history.png')
    
    # Evaluate model on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
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
    plt.title('Load Forecasting: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Active Users')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_results.png')
    
    # Save model and return results
    model.save('gru_attention_forecast_model.h5')
    
    return {
        'model': model,
        'best_hyperparams': best_hyperparams,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'scaler': scaler,
        'input_shape': X_train.shape[1:]
    }


def forecast_future_load(model, last_sequence, scaler, n_steps=24):
    """Forecast future load using trained model"""
    future_preds = []
    current_seq = last_sequence.copy()
    
    for i in range(n_steps):
        # Predict next step
        next_pred = model.predict(current_seq.reshape(1, *current_seq.shape))
        
        # Add prediction to result
        future_preds.append(next_pred[0, 0])
        
        # Update sequence
        # Shift sequence and add prediction
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, 0] = next_pred[0, 0]
    
    # Create dummy array for inverse transform
    dummy = np.zeros((len(future_preds), current_seq.shape[1]))
    dummy[:, 0] = future_preds
    
    # Inverse transform predictions
    future_preds_inv = scaler.inverse_transform(dummy)[:, 0]
    
    return future_preds_inv


def main():
    """Main function to train load forecasting model"""
    # Load processed IMFs
    imfs, params = load_processed_imfs('processed_imfs.npz')
    
    # Load hourly data
    # Assuming this is generated from preprocessing step
    data, hourly_data = iwoa_vmd.load_shanghai_telecom_data('data/DataSet/combined_output.csv')
    
    # Prepare data for forecasting
    X_data, scaler = prepare_forecast_data(imfs, hourly_data)
    
    # Train forecasting model
    results = train_forecasting_model(X_data, scaler)
    
    # Save results for Java integration
    tf.keras.models.save_model(results['model'], 'forecast_model.keras')  # Recommended
    print(f"Model saved with best hyperparameters: {results['best_hyperparams']}")
    print(f"Test MAE: {results['test_mae']}")
    
    # Forecast future load (next 24 hours)
    # Get last known sequence
    X, _ = create_sequences(X_data, 24, 6)
    last_sequence = X[-1]
    
    future_load = forecast_future_load(results['model'], last_sequence, scaler)
    
    # Save future load forecast
    np.save('future_load_forecast.npy', future_load)
    print(f"Forecasted load for next 24 hours: {future_load}")
    
    return results


if __name__ == "__main__":
    main()