import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import json
import random
import time
from scipy.optimize import minimize, linprog
from collections import defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class TaskOffloadingServiceMigration:
    """
    Simulation framework for task offloading and service migration in edge computing
    Based on Stackelberg game for task offloading and Two-Stage TIGO with Min-Max Fairness
    """
    def __init__(self, forecast_file='results/combined_output.csv', 
                 models_dir='models', results_dir='results'):
        self.forecast_file = forecast_file
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.edge_nodes = []
        self.forecasts = None
        self.edge_capacities = {}
        self.edge_compute_power = {}
        self.edge_energy_efficiency = {}
        self.edge_bandwidth = {}
        self.service_requirements = {}
        self.user_task_requirements = {}
        self.service_placements = {}
        self.task_assignments = {}
        self.stackelberg_prices = {}
        self.migration_decisions = {}
        self.simulation_results = {}
        
        # Ensure the simulation results directory exists
        if not os.path.exists(f"{self.results_dir}/simulation"):
            os.makedirs(f"{self.results_dir}/simulation")
    
    def load_forecasts(self):
        """Load forecasted edge node loads"""
        if os.path.exists(self.forecast_file):
            self.forecasts = pd.read_csv(self.forecast_file, parse_dates=['timestamp'])
            self.edge_nodes = self.forecasts['edge_node'].unique()
            print(f"Loaded forecasts for {len(self.edge_nodes)} edge nodes")
            return True
        else:
            print(f"Error: Forecast file not found at {self.forecast_file}")
            return False
    
    def setup_simulation_parameters(self):
        """Initialize simulation parameters for edge nodes and services"""
        # Assign random capacities and characteristics to edge nodes
        for edge_node in self.edge_nodes:
            # Higher number = more capacity/power
            self.edge_capacities[edge_node] = random.randint(800, 1500)  # Max concurrent users
            self.edge_compute_power[edge_node] = random.uniform(0.8, 2.0)  # Relative compute power
            self.edge_energy_efficiency[edge_node] = random.uniform(0.6, 0.95)  # Energy efficiency
            self.edge_bandwidth[edge_node] = random.uniform(50, 200)  # Mbps

        # Define services with different resource requirements
        services = ['streaming', 'gaming', 'ar_vr', 'iot_processing', 'voice_chat']
        
        for service in services:
            # Resource requirements for different services
            if service == 'streaming':
                cpu = random.uniform(0.1, 0.2)  # CPU requirements per user
                memory = random.uniform(50, 100)  # MB per user
                bandwidth = random.uniform(3, 8)  # Mbps per user
                latency_sensitive = 0.6  # Medium sensitivity
            elif service == 'gaming':
                cpu = random.uniform(0.2, 0.4)
                memory = random.uniform(200, 400)
                bandwidth = random.uniform(5, 15)
                latency_sensitive = 0.9  # High sensitivity
            elif service == 'ar_vr':
                cpu = random.uniform(0.3, 0.6)
                memory = random.uniform(300, 600)
                bandwidth = random.uniform(15, 30)
                latency_sensitive = 0.95  # Very high sensitivity
            elif service == 'iot_processing':
                cpu = random.uniform(0.05, 0.1)
                memory = random.uniform(20, 50)
                bandwidth = random.uniform(0.5, 2)
                latency_sensitive = 0.4  # Low sensitivity
            else:  # voice_chat
                cpu = random.uniform(0.05, 0.15)
                memory = random.uniform(30, 70)
                bandwidth = random.uniform(1, 3)
                latency_sensitive = 0.75  # Medium-high sensitivity
                
            self.service_requirements[service] = {
                'cpu': cpu,
                'memory': memory,
                'bandwidth': bandwidth,
                'latency_sensitive': latency_sensitive
            }
            
        # Distribute services across edge nodes (initial placement)
        for service in services:
            eligible_nodes = random.sample(list(self.edge_nodes), k=min(len(self.edge_nodes), 3))
            self.service_placements[service] = eligible_nodes
            
        print(f"Simulation parameters set up for {len(self.edge_nodes)} edge nodes and {len(services)} services")
        return True
    
    def generate_user_tasks(self, time_step):
        """Generate tasks based on forecasted user load at given time step"""
        # Filter forecasts for the specific time step
        current_forecasts = self.forecasts[self.forecasts['timestamp'] == self.forecasts['timestamp'].unique()[time_step]]
        
        tasks = []
        task_id = 0
        
        for _, row in current_forecasts.iterrows():
            edge_node = row['edge_node']
            user_count = int(row['predicted_active_users'])
            
            # Generate tasks for each user (simplified - in reality not all users generate tasks)
            active_users = min(user_count, self.edge_capacities[edge_node])
            task_count = int(active_users * random.uniform(0.3, 0.7))  # Not all users have tasks
            
            for _ in range(task_count):
                # Randomly assign a service type to the task
                service = random.choice(list(self.service_requirements.keys()))
                
                # Generate task requirements based on service type with some variability
                service_req = self.service_requirements[service]
                task_req = {
                    'id': task_id,
                    'service': service,
                    'origin_node': edge_node,
                    'cpu': service_req['cpu'] * random.uniform(0.8, 1.2),
                    'memory': service_req['memory'] * random.uniform(0.8, 1.2),
                    'bandwidth': service_req['bandwidth'] * random.uniform(0.8, 1.2),
                    'latency_req': service_req['latency_sensitive'] * random.uniform(0.8, 1.2),
                    'deadline': random.uniform(50, 200),  # Task deadline in ms
                    'priority': random.uniform(0.1, 1.0)  # Task priority
                }
                
                tasks.append(task_req)
                task_id += 1
                
        self.user_task_requirements[time_step] = tasks
        print(f"Generated {len(tasks)} tasks for time step {time_step}")
        return tasks
    
    def calculate_latency(self, origin_node, target_node):
        """Calculate network latency between two edge nodes"""
        # Get coordinates for both nodes
        origin_data = self.forecasts[self.forecasts['edge_node'] == origin_node].iloc[0]
        target_data = self.forecasts[self.forecasts['edge_node'] == target_node].iloc[0]
        
        # Calculate Euclidean distance (simplified)
        distance = np.sqrt((origin_data['latitude'] - target_data['latitude'])**2 + 
                          (origin_data['longitude'] - target_data['longitude'])**2)
        
        # Convert distance to latency (simplified model)
        # Closer nodes have lower latency; add baseline latency
        base_latency = 5  # ms
        distance_factor = 50  # ms per unit distance
        
        latency = base_latency + (distance * distance_factor)
        return latency
    
    def calculate_node_load(self, node, time_step):
        """Calculate current load on a node at given time step"""
        if time_step in self.task_assignments:
            # Count tasks assigned to this node
            node_tasks = [task for task in self.task_assignments[time_step] 
                         if task['assigned_node'] == node]
            
            # Calculate resource usage
            cpu_usage = sum(task['task']['cpu'] for task in node_tasks)
            memory_usage = sum(task['task']['memory'] for task in node_tasks)
            bandwidth_usage = sum(task['task']['bandwidth'] for task in node_tasks)
            
            # Normalize by node capacity
            cpu_load = cpu_usage / self.edge_compute_power[node]
            memory_load = memory_usage / (self.edge_capacities[node] * 10)  # Assuming 10MB per capacity unit
            bandwidth_load = bandwidth_usage / self.edge_bandwidth[node]
            
            # Calculate weighted load
            load = 0.4 * cpu_load + 0.3 * memory_load + 0.3 * bandwidth_load
            
            return load
        else:
            # No tasks assigned yet
            return 0.0

    def run_stackelberg_orchestration(self, time_step):
        """
        Implement Stackelberg game for task offloading
        - Leader (orchestrator) sets prices based on edge node capacities
        - Followers (tasks) choose nodes to maximize their utility
        """
        tasks = self.user_task_requirements[time_step]
        
        # Leader phase: Set prices based on current load and capacity
        self.stackelberg_prices[time_step] = {}
        node_loads = {}
        
        for node in self.edge_nodes:
            # Get forecasted load for this node
            node_forecast = self.forecasts[(self.forecasts['edge_node'] == node) & 
                                          (self.forecasts['timestamp'] == self.forecasts['timestamp'].unique()[time_step])]
            
            if len(node_forecast) > 0:
                forecasted_load = node_forecast.iloc[0]['predicted_active_users'] / self.edge_capacities[node]
                
                # Set price based on load and compute power (higher load = higher price)
                # Lower compute power = higher price
                base_price = 0.5
                load_factor = 1.5
                compute_factor = 0.8
                
                price = base_price * (1 + (load_factor * forecasted_load)) / (compute_factor * self.edge_compute_power[node])
                
                self.stackelberg_prices[time_step][node] = price
                node_loads[node] = forecasted_load
            else:
                self.stackelberg_prices[time_step][node] = 1.0  # Default price
                node_loads[node] = 0.5  # Default load
        
        # Follower phase: Assign tasks to nodes based on utility maximization
        assignments = []
        
        for task in tasks:
            # Find nodes that offer the service required by the task
            eligible_nodes = self.service_placements[task['service']]
            
            best_node = None
            best_utility = float('-inf')
            
            for node in eligible_nodes:
                # Calculate utility for each node
                
                # Calculate latency based on origin and target
                latency = self.calculate_latency(task['origin_node'], node)
                normalized_latency = min(1.0, latency / task['deadline'])
                
                # Utility components
                price_factor = -1 * self.stackelberg_prices[time_step][node]
                latency_factor = -2 * normalized_latency * task['latency_req']
                power_factor = self.edge_compute_power[node]
                
                utility = (0.4 * price_factor + 
                          0.4 * latency_factor + 
                          0.2 * power_factor)
                
                if utility > best_utility:
                    best_utility = utility
                    best_node = node
            
            # Assign task to the best node
            if best_node:
                assignments.append({
                    'task': task,
                    'assigned_node': best_node,
                    'utility': best_utility,
                    'price': self.stackelberg_prices[time_step][best_node]
                })
            else:
                # No suitable node found, assign to origin as fallback
                assignments.append({
                    'task': task,
                    'assigned_node': task['origin_node'],
                    'utility': -999,  # Very low utility
                    'price': self.stackelberg_prices[time_step].get(task['origin_node'], 1.0)
                })
        
        self.task_assignments[time_step] = assignments
        
        # Calculate metrics for this time step
        total_tasks = len(assignments)
        offloaded_tasks = sum(1 for a in assignments if a['assigned_node'] != a['task']['origin_node'])
        avg_utility = sum(a['utility'] for a in assignments) / total_tasks if total_tasks > 0 else 0
        
        metrics = {
            'total_tasks': total_tasks,
            'offloaded_tasks': offloaded_tasks,
            'offloading_ratio': offloaded_tasks / total_tasks if total_tasks > 0 else 0,
            'avg_utility': avg_utility
        }
        
        print(f"Stackelberg orchestration for time step {time_step} - " +
              f"Offloaded {offloaded_tasks}/{total_tasks} tasks " +
              f"({metrics['offloading_ratio']:.2%})")
        
        return metrics
    
    def calculate_migration_cost(self, service, source_node, target_node):
        """Calculate cost of migrating a service from source to target node"""
        # Migration cost components:
        # 1. Data transfer cost (based on service memory requirements and distance)
        # 2. Service downtime cost (based on service sensitivity)
        
        # Estimate distance between nodes
        source_data = self.forecasts[self.forecasts['edge_node'] == source_node].iloc[0]
        target_data = self.forecasts[self.forecasts['edge_node'] == target_node].iloc[0]
        
        distance = np.sqrt((source_data['latitude'] - target_data['latitude'])**2 + 
                          (source_data['longitude'] - target_data['longitude'])**2)
        
        # Calculate data transfer cost
        service_memory = self.service_requirements[service]['memory']
        data_transfer_cost = service_memory * distance * 0.001  # Cost per MB per distance unit
        
        # Calculate downtime cost
        latency_sensitivity = self.service_requirements[service]['latency_sensitive']
        downtime_cost = latency_sensitivity * 50  # Base downtime cost
        
        total_cost = data_transfer_cost + downtime_cost
        return total_cost
    
    def run_tigo_service_migration(self, time_step):
        """
        Two-Stage TIGO approach for service migration with Min-Max Fairness
        Stage 1: Determine if migration is needed based on load imbalance
        Stage 2: Execute migration using Min-Max Fairness to balance load
        """
        if time_step == 0:
            # No migration needed for first time step
            return {"migrations": 0, "load_variance_before": 0, "load_variance_after": 0}
        
        # Stage 1: Determine if migration is needed
        node_loads = {}
        for node in self.edge_nodes:
            # Get assigned tasks for this node
            node_load = self.calculate_node_load(node, time_step)
            node_loads[node] = node_load
        
        # Calculate load imbalance
        load_values = list(node_loads.values())
        load_variance = np.var(load_values)
        load_mean = np.mean(load_values)
        
        # Migration is needed if variance is above threshold or max load is too high
        migration_threshold = 0.05  # Variance threshold
        max_load_threshold = 0.8  # Maximum desired load
        
        max_load = max(load_values)
        migration_needed = (load_variance > migration_threshold or max_load > max_load_threshold)
        
        migrations = []
        
        if migration_needed:
            # Stage 2: Execute migration using Min-Max Fairness
            
            # Map services to nodes based on current task assignments
            service_node_counts = defaultdict(lambda: defaultdict(int))
            
            # Count services on each node
            for assignment in self.task_assignments[time_step]:
                service = assignment['task']['service']
                node = assignment['assigned_node']
                service_node_counts[service][node] += 1
            
            # Identify overloaded and underloaded nodes
            overloaded_nodes = [node for node, load in node_loads.items() if load > load_mean * 1.2]
            underloaded_nodes = [node for node, load in node_loads.items() if load < load_mean * 0.8]
            
            # For each service, consider migration from overloaded to underloaded nodes
            for service in self.service_placements.keys():
                for source_node in overloaded_nodes:
                    # Only consider migrating if service is present and used on source node
                    if source_node in self.service_placements[service] and service_node_counts[service][source_node] > 0:
                        for target_node in underloaded_nodes:
                            # Check if target node doesn't already have this service
                            if target_node not in self.service_placements[service]:
                                # Calculate migration cost
                                migration_cost = self.calculate_migration_cost(service, source_node, target_node)
                                
                                # Calculate benefit (load reduction)
                                service_load = (service_node_counts[service][source_node] * 
                                               self.service_requirements[service]['cpu'] / 
                                               self.edge_compute_power[source_node])
                                
                                # Potential load reduction (benefit)
                                benefit = service_load
                                
                                # Migrate if benefit outweighs cost
                                if benefit > migration_cost * 0.1:  # Threshold factor
                                    # Execute migration
                                    migrations.append({
                                        'service': service,
                                        'from_node': source_node,
                                        'to_node': target_node,
                                        'cost': migration_cost,
                                        'benefit': benefit
                                    })
                                    
                                    # Update service placements
                                    if target_node not in self.service_placements[service]:
                                        self.service_placements[service].append(target_node)
                                    
                                    # Update node loads to reflect migration
                                    node_loads[source_node] -= service_load
                                    node_loads[target_node] += service_load
                                    
                                    # Limit migrations per iteration
                                    if len(migrations) >= 5:
                                        break
                            
                        if len(migrations) >= 5:
                            break
                    
                if len(migrations) >= 5:
                    break
        
        # Calculate new load variance after migrations
        new_load_values = list(node_loads.values())
        new_load_variance = np.var(new_load_values)
        
        self.migration_decisions[time_step] = migrations
        
        metrics = {
            "migrations": len(migrations),
            "load_variance_before": load_variance,
            "load_variance_after": new_load_variance,
            "load_reduction": (load_variance - new_load_variance) / load_variance if load_variance > 0 else 0
        }
        
        print(f"TIGO migration for time step {time_step} - " +
              f"Executed {len(migrations)} migrations, " +
              f"Load variance reduced by {metrics['load_reduction']:.2%}")
        
        return metrics
    
    def run_simulation(self, num_time_steps=12):
        """Run full simulation for specified number of time steps"""
        if not self.load_forecasts():
            return False
        
        if not self.setup_simulation_parameters():
            return False
        
        print(f"Starting simulation for {num_time_steps} time steps...")
        simulation_metrics = []
        
        for t in range(num_time_steps):
            print(f"\n--- Time Step {t} ---")
            
            # Generate tasks based on forecasted user load
            self.generate_user_tasks(t)
            
            # Run Stackelberg orchestration for task offloading
            offload_metrics = self.run_stackelberg_orchestration(t)
            
            # Run TIGO service migration
            migration_metrics = self.run_tigo_service_migration(t)
            
            # Record metrics for this time step
            time_metrics = {
                'time_step': t,
                'timestamp': self.forecasts['timestamp'].unique()[t],
                **offload_metrics,
                **migration_metrics
            }
            
            simulation_metrics.append(time_metrics)
        
        # Save results
        self.simulation_results = simulation_metrics
        
        return True
    
    def analyze_results(self):
        """Analyze and visualize simulation results"""
        if not self.simulation_results:
            print("No simulation results to analyze")
            return False
        
        results_df = pd.DataFrame(self.simulation_results)
        
        # Save results to CSV
        results_df.to_csv(f"{self.results_dir}/simulation/simulation_metrics.csv", index=False)
        
        # Plot key metrics
        plt.figure(figsize=(15, 12))
        
        # 1. Task offloading ratio
        plt.subplot(3, 2, 1)
        plt.plot(results_df['time_step'], results_df['offloading_ratio'], marker='o', linestyle='-')
        plt.title('Task Offloading Ratio')
        plt.xlabel('Time Step')
        plt.ylabel('Offloading Ratio')
        plt.grid(True)
        
        # 2. Average utility
        plt.subplot(3, 2, 2)
        plt.plot(results_df['time_step'], results_df['avg_utility'], marker='o', linestyle='-')
        plt.title('Average Task Utility')
        plt.xlabel('Time Step')
        plt.ylabel('Utility')
        plt.grid(True)
        
        # 3. Number of migrations
        plt.subplot(3, 2, 3)
        plt.bar(results_df['time_step'], results_df['migrations'])
        plt.title('Service Migrations')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Migrations')
        plt.grid(True)
        
        # 4. Load variance reduction
        plt.subplot(3, 2, 4)
        plt.plot(results_df['time_step'], results_df['load_variance_before'], marker='o', label='Before Migration')
        plt.plot(results_df['time_step'], results_df['load_variance_after'], marker='x', label='After Migration')
        plt.title('Load Variance Before and After Migration')
        plt.xlabel('Time Step')
        plt.ylabel('Load Variance')
        plt.legend()
        plt.grid(True)
        
        # 5. Total tasks
        plt.subplot(3, 2, 5)
        plt.plot(results_df['time_step'], results_df['total_tasks'], marker='o', linestyle='-')
        plt.title('Total Tasks')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Tasks')
        plt.grid(True)
        
        # 6. Load reduction percentage
        plt.subplot(3, 2, 6)
        plt.plot(results_df['time_step'], results_df['load_reduction'] * 100, marker='o', linestyle='-')
        plt.title('Load Variance Reduction (%)')
        plt.xlabel('Time Step')
        plt.ylabel('Reduction (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/simulation/simulation_metrics.png")
        
        # Create a node load heatmap
        self.plot_node_load_heatmap()
        
        # Calculate efficiency metrics
        efficiency_metrics = self.calculate_efficiency_metrics()
        
        # Save efficiency metrics
        with open(f"{self.results_dir}/simulation/efficiency_metrics.json", 'w') as f:
            json.dump(efficiency_metrics, f, indent=4)
        
        print("Simulation analysis completed and saved to results/simulation directory")
        return True
    
    def plot_node_load_heatmap(self):
        """Create a heatmap of node loads over time"""
        # Get unique time steps and nodes
        time_steps = sorted(list(self.task_assignments.keys()))
        nodes = sorted(list(self.edge_nodes))
        
        # Create a matrix of loads
        load_matrix = np.zeros((len(nodes), len(time_steps)))
        
        for t_idx, t in enumerate(time_steps):
            for n_idx, node in enumerate(nodes):
                load_matrix[n_idx, t_idx] = self.calculate_node_load(node, t)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(load_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Node Load')
        plt.title('Edge Node Load Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Edge Node')
        plt.yticks(range(len(nodes)), nodes)
        plt.xticks(range(len(time_steps)), time_steps)
        plt.savefig(f"{self.results_dir}/simulation/node_load_heatmap.png")
        
        return True
    
    def calculate_efficiency_metrics(self):
        """Calculate various efficiency metrics for the simulation"""
        metrics = {}
        
        # 1. Average resource utilization
        node_loads = []
        for t in self.task_assignments.keys():
            for node in self.edge_nodes:
                node_loads.append(self.calculate_node_load(node, t))
        
        metrics['avg_resource_utilization'] = np.mean(node_loads)
        metrics['max_resource_utilization'] = np.max(node_loads)
        metrics['min_resource_utilization'] = np.min(node_loads)
        metrics['std_resource_utilization'] = np.std(node_loads)
        
        # 2. Service migration statistics
        total_migrations = sum(len(migrations) for migrations in self.migration_decisions.values())
        metrics['total_migrations'] = total_migrations
        
        if total_migrations > 0:
            total_cost = sum(m['cost'] for t in self.migration_decisions 
                            for m in self.migration_decisions[t])
            total_benefit = sum(m['benefit'] for t in self.migration_decisions 
                               for m in self.migration_decisions[t])
            
            metrics['avg_migration_cost'] = total_cost / total_migrations
            metrics['avg_migration_benefit'] = total_benefit / total_migrations
            metrics['benefit_cost_ratio'] = total_benefit / total_cost if total_cost > 0 else float('inf')
        
        # 3. Task offloading statistics
        total_tasks = sum(len(tasks) for tasks in self.user_task_requirements.values())
        offloaded_tasks = sum(sum(1 for a in self.task_assignments[t] 
                                 if a['assigned_node'] != a['task']['origin_node'])
                             for t in self.task_assignments)
        
        metrics['total_tasks'] = total_tasks
        metrics['offloaded_tasks'] = offloaded_tasks
        metrics['offloading_ratio'] = offloaded_tasks / total_tasks if total_tasks > 0 else 0
        
        # 4. Load balancing metrics
        load_variance_reduction = np.mean([
            (m['load_variance_before'] - m['load_variance_after']) / m['load_variance_before'] 
            if m['load_variance_before'] > 0 else 0
            for m in self.simulation_results if 'load_variance_before' in m
        ])
        
        metrics['avg_load_variance_reduction'] = load_variance_reduction
        
        return metrics

def main():
    """Main function to run the simulation"""
    start_time = time.time()
    
    # Initialize simulation
    sim = TaskOffloadingServiceMigration()
    
    # Run simulation for 12 time steps
    if sim.run_simulation(num_time_steps=12):
        # Analyze results
        sim.analyze_results()
        
        # Print execution time
        execution_time = time.time() - start_time
        print(f"Simulation completed in {execution_time:.2f} seconds")
    else:
        print("Simulation failed to run")

if __name__ == "__main__":
    main()