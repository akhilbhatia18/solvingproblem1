import os
import sys
import pandas as pd
import time

# Import our modules
from offloading_migration import TaskOffloadingServiceMigration
from simulation_analyzer import SimulationAnalyzer
from generate_test_data import generate_test_dataset

def run_complete_test(num_edge_nodes=10, num_time_steps=12, analysis=True):
    """
    Run a complete test of the offloading and migration simulation workflow
    
    Parameters:
    -----------
    num_edge_nodes : int
        Number of edge nodes to simulate
    num_time_steps : int
        Number of time steps for the simulation
    analysis : bool
        Whether to run analysis after simulation
    """
    print("=" * 50)
    print("TASK OFFLOADING AND SERVICE MIGRATION SIMULATION TEST")
    print("=" * 50)
    
    # 1. Create results directory structure
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/simulation", exist_ok=True)
    
    # 2. Generate test dataset
    print("\n[1/3] Generating test dataset...")
    forecast_file = 'results/all_locations_forecast.csv'
    generate_test_dataset(output_file=forecast_file, 
                          num_edge_nodes=num_edge_nodes, 
                          num_time_steps=max(num_time_steps, 24))  # Generate at least 24 time steps
    
    # 3. Run simulation
    print("\n[2/3] Running simulation...")
    start_time = time.time()
    sim = TaskOffloadingServiceMigration(forecast_file=forecast_file)
    
    if sim.run_simulation(num_time_steps=num_time_steps):
        sim.analyze_results()
        simulation_time = time.time() - start_time
        print(f"\nSimulation completed in {simulation_time:.2f} seconds")
    else:
        print("\nSimulation failed!")
        return False
    
    # 4. Run analysis (optional)
    if analysis:
        print("\n[3/3] Running advanced analysis...")
        analyzer = SimulationAnalyzer(results_dir='results/simulation')
        if analyzer.analyze_all():
            print("\nAnalysis completed successfully")
        else:
            print("\nAnalysis failed!")
            return False
    
    print("\n" + "=" * 50)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 50)
    return True

if __name__ == "__main__":
    # Parse command line arguments
    num_nodes = 50
    num_timesteps = 24
    run_analysis = True
    
    if len(sys.argv) > 1:
        num_nodes = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_timesteps = int(sys.argv[2])
    if len(sys.argv) > 3:
        run_analysis = sys.argv[3].lower() in ['true', 't', '1', 'yes', 'y']
    
    run_complete_test(num_edge_nodes=num_nodes, 
                     num_time_steps=num_timesteps,
                     analysis=run_analysis)