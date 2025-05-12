import pandas as pd
import numpy as np
import os
import datetime

def generate_test_dataset(output_file='results/all_locations_forecast.csv', 
                          num_edge_nodes=10, 
                          num_time_steps=24):
    """
    Generate a synthetic edge node forecast dataset for testing the simulation
    
    Parameters:
    -----------
    output_file : str
        Path to save the CSV file
    num_edge_nodes : int
        Number of edge nodes to simulate
    num_time_steps : int
        Number of time steps to generate forecasts for
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a base timestamp
    base_time = datetime.datetime(2025, 4, 1, 0, 0)
    
    # Generate edge node IDs and locations
    edge_nodes = [f"edge_node_{i}" for i in range(1, num_edge_nodes + 1)]
    
    # Generate random coordinates (latitude and longitude) for each edge node
    # Using a range that approximates coordinates in a region
    np.random.seed(42)  # For reproducibility
    latitudes = np.random.uniform(35.0, 42.0, num_edge_nodes)  # Example: US East Coast range
    longitudes = np.random.uniform(-80.0, -70.0, num_edge_nodes)
    
    # Create forecasts data
    data = []
    
    for t in range(num_time_steps):
        timestamp = base_time + datetime.timedelta(hours=t)
        
        # Create time-dependent patterns
        # 1. Daily pattern (more users during daytime)
        hour_of_day = timestamp.hour
        day_factor = 0.5 + 0.5 * np.sin(np.pi * hour_of_day / 12)  # 0.5-1.5 multiplier
        
        for i, node in enumerate(edge_nodes):
            # Base load (different for each node)
            base_load = np.random.uniform(50, 200)
            
            # Add time variation
            time_variation = day_factor * base_load
            
            # Add some random noise
            noise = np.random.normal(0, base_load * 0.1)
            
            # Calculate predicted active users
            predicted_users = max(0, int(time_variation + noise))
            
            # Add row to data
            data.append({
                'timestamp': timestamp,
                'edge_node': node,
                'latitude': latitudes[i],
                'longitude': longitudes[i],
                'predicted_active_users': predicted_users
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated test dataset with {num_edge_nodes} edge nodes and {num_time_steps} time steps")
    print(f"Data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate test dataset
    generate_test_dataset()