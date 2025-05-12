import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats

class SimulationAnalyzer:
    """
    Analyze and compare different task offloading and service migration strategies
    """
    def __init__(self, results_dir='results/simulation'):
        self.results_dir = results_dir
        self.metrics_file = f"{results_dir}/simulation_metrics.csv"
        self.efficiency_file = f"{results_dir}/efficiency_metrics.json"
        self.metrics_df = None
        self.efficiency_metrics = None
        
    def load_data(self):
        """Load simulation results data"""
        if os.path.exists(self.metrics_file):
            self.metrics_df = pd.read_csv(self.metrics_file)
            print(f"Loaded metrics data with {len(self.metrics_df)} time steps")
        else:
            print(f"Error: Metrics file not found at {self.metrics_file}")
            return False
        
        if os.path.exists(self.efficiency_file):
            with open(self.efficiency_file, 'r') as f:
                self.efficiency_metrics = json.load(f)
            print("Loaded efficiency metrics")
        else:
            print(f"Warning: Efficiency metrics file not found at {self.efficiency_file}")
        
        return True
    
    def generate_advanced_plots(self):
        """Create advanced analysis plots"""
        if self.metrics_df is None:
            print("No data loaded. Please call load_data() first.")
            return False
        
        # Create output directory
        analysis_dir = f"{self.results_dir}/analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        # Set seaborn style
        sns.set(style="whitegrid")
        
        # 1. Combined offloading and migration visualization
        plt.figure(figsize=(14, 8))
        
        ax1 = plt.gca()
        
        # Offloading ratio line
        ax1.plot(self.metrics_df['time_step'], self.metrics_df['offloading_ratio'], 
                marker='o', linestyle='-', color='blue', label='Offloading Ratio')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Offloading Ratio', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Migration count bars on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(self.metrics_df['time_step'], self.metrics_df['migrations'], 
               alpha=0.5, color='red', label='Migrations')
        ax2.set_ylabel('Number of Migrations', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Task Offloading and Service Migration Over Time')
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        plt.savefig(f"{analysis_dir}/offload_migration_combined.png")
        
        # 2. Load Variance Reduction Analysis
        plt.figure(figsize=(12, 7))
        
        # Plot the percentage reduction
        plt.bar(self.metrics_df['time_step'], self.metrics_df['load_reduction'] * 100,
               color='green', alpha=0.7)
        
        # Overlay line for before and after variance
        plt.plot(self.metrics_df['time_step'], self.metrics_df['load_variance_before'], 
                'o-', color='red', label='Before Migration')
        plt.plot(self.metrics_df['time_step'], self.metrics_df['load_variance_after'], 
                'o-', color='blue', label='After Migration')
        
        plt.title('Load Variance Reduction Analysis')
        plt.xlabel('Time Step')
        plt.ylabel('Load Variance / Reduction %')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{analysis_dir}/load_variance_reduction.png")
        
        # 3. Task Utility vs Node Load Correlation
        # Calculate average node load for each time step
        avg_utility = self.metrics_df['avg_utility']
        load_variance = self.metrics_df['load_variance_before']
        
        plt.figure(figsize=(10, 8))
        plt.scatter(load_variance, avg_utility, c=self.metrics_df['time_step'], 
                   cmap='viridis', s=100, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(load_variance, avg_utility, 1)
        p = np.poly1d(z)
        plt.plot(load_variance, p(load_variance), "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr, _ = stats.pearsonr(load_variance, avg_utility)
        plt.title(f'Task Utility vs Load Variance (Correlation: {corr:.2f})')
        plt.xlabel('Load Variance')
        plt.ylabel('Average Task Utility')
        plt.colorbar(label='Time Step')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{analysis_dir}/utility_load_correlation.png")
        
        # 4. Efficiency over time
        plt.figure(figsize=(12, 8))
        
        # Calculate efficiency ratio (benefit/cost) if available
        if 'migrations' in self.metrics_df and np.sum(self.metrics_df['migrations']) > 0:
            # Create a new column for efficiency if migrations happened
            # Use the ratio of load reduction to migration count as a proxy for efficiency
            self.metrics_df['efficiency_proxy'] = self.metrics_df['load_reduction'] / \
                                                 (self.metrics_df['migrations'] + 0.001)  # Avoid div by zero
            
            plt.plot(self.metrics_df['time_step'], self.metrics_df['efficiency_proxy'], 
                    'o-', color='purple', linewidth=2)
            plt.title('Migration Efficiency Over Time (Load Reduction per Migration)')
            plt.xlabel('Time Step')
            plt.ylabel('Efficiency (Load Reduction / Migration)')
            plt.grid(True)
            plt.savefig(f"{analysis_dir}/migration_efficiency.png")
        
        # 5. Violin plot of task distribution across nodes
        if self.metrics_df['total_tasks'].sum() > 0:
            plt.figure(figsize=(12, 8))
            
            # Create time step groups for visualization
            time_groups = []
            tasks_per_step = []
            
            for t in self.metrics_df['time_step']:
                time_str = f"Step {t}"
                task_count = self.metrics_df.loc[self.metrics_df['time_step'] == t, 'total_tasks'].values[0]
                times = [time_str] * task_count
                time_groups.extend(times)
                tasks_per_step.extend(range(task_count))
            
            # Create a violin plot if we have enough data
            if len(time_groups) > 10:  # Only create if we have enough tasks
                violin_data = pd.DataFrame({
                    'Time Step': time_groups,
                    'Task Index': tasks_per_step
                })
                
                sns.violinplot(x='Time Step', y='Task Index', data=violin_data)
                plt.title('Task Distribution Over Time')
                plt.xlabel('Time Step')
                plt.ylabel('Task Index')
                plt.savefig(f"{analysis_dir}/task_distribution.png")
        
        print(f"Advanced plots generated and saved to {analysis_dir}")
        return True
    
    def calculate_comparative_metrics(self):
        """Calculate comparative metrics for evaluating the strategies"""
        if self.metrics_df is None or self.efficiency_metrics is None:
            print("No data loaded. Please call load_data() first.")
            return False
        
        comparative_metrics = {}
        
        # 1. Average offloading ratio
        comparative_metrics['avg_offloading_ratio'] = self.metrics_df['offloading_ratio'].mean()
        
        # 2. Migration frequency
        comparative_metrics['migration_frequency'] = self.metrics_df['migrations'].mean()
        
        # 3. Load balancing effectiveness
        comparative_metrics['load_balance_improvement'] = self.metrics_df['load_reduction'].mean() * 100  # percentage
        
        # 4. Resource utilization
        if 'avg_resource_utilization' in self.efficiency_metrics:
            comparative_metrics['resource_utilization'] = self.efficiency_metrics['avg_resource_utilization']
            comparative_metrics['resource_utilization_std'] = self.efficiency_metrics['std_resource_utilization']
        
        # 5. Migration efficiency
        if 'benefit_cost_ratio' in self.efficiency_metrics:
            comparative_metrics['migration_efficiency'] = self.efficiency_metrics['benefit_cost_ratio']
        
        # Save comparative metrics to file
        analysis_dir = f"{self.results_dir}/analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        with open(f"{analysis_dir}/comparative_metrics.json", 'w') as f:
            json.dump(comparative_metrics, f, indent=4)
            
        print(f"Comparative metrics calculated and saved to {analysis_dir}/comparative_metrics.json")
        return comparative_metrics
    
    def create_summary_report(self):
        """Create a summary report of the simulation results"""
        if self.metrics_df is None:
            print("No data loaded. Please call load_data() first.")
            return False
            
        # Calculate key statistics
        comparative_metrics = self.calculate_comparative_metrics()
        
        # Create output directory
        analysis_dir = f"{self.results_dir}/analysis"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
            
        # Create a summary report
        with open(f"{analysis_dir}/summary_report.md", 'w') as f:
            f.write("# Task Offloading and Service Migration Simulation Summary\n\n")
            
            f.write("## Overview\n")
            f.write(f"- Total time steps: {len(self.metrics_df)}\n")
            f.write(f"- Total tasks generated: {self.metrics_df['total_tasks'].sum()}\n")
            f.write(f"- Total offloaded tasks: {self.metrics_df['offloaded_tasks'].sum()}\n")
            f.write(f"- Total service migrations: {self.metrics_df['migrations'].sum()}\n\n")
            
            f.write("## Performance Metrics\n")
            f.write(f"- Average offloading ratio: {comparative_metrics.get('avg_offloading_ratio', 'N/A'):.2%}\n")
            f.write(f"- Average migrations per time step: {comparative_metrics.get('migration_frequency', 'N/A'):.2f}\n")
            f.write(f"- Load balance improvement: {comparative_metrics.get('load_balance_improvement', 'N/A'):.2f}%\n")
            f.write(f"- Resource utilization: {comparative_metrics.get('resource_utilization', 'N/A'):.2f}\n")
            f.write(f"- Migration benefit-cost ratio: {comparative_metrics.get('migration_efficiency', 'N/A'):.2f}\n\n")
            
            f.write("## Time Step Analysis\n")
            f.write("| Time Step | Tasks | Offloaded Tasks | Migrations | Load Reduction | Avg Utility |\n")
            f.write("|-----------|-------|----------------|------------|---------------|------------|\n")
            
            for _, row in self.metrics_df.iterrows():
                f.write(f"| {int(row['time_step'])} | {int(row['total_tasks'])} | {int(row['offloaded_tasks'])} | ")
                f.write(f"{int(row['migrations'])} | {row['load_reduction']:.2%} | {row['avg_utility']:.3f} |\n")
            
            f.write("\n## Conclusion\n")
            f.write("This simulation evaluated the effectiveness of the Stackelberg game approach for task offloading ")
            f.write("combined with the Two-Stage TIGO method for service migration. ")
            
            # Add conclusion based on metrics
            if comparative_metrics.get('avg_offloading_ratio', 0) > 0.5:
                f.write("The task offloading strategy was highly effective, with more than half of all tasks ")
                f.write("being offloaded from their origin nodes. ")
            else:
                f.write("The task offloading strategy showed moderate effectiveness, with less than half of the tasks ")
                f.write("being offloaded from their origin nodes. ")
                
            if comparative_metrics.get('load_balance_improvement', 0) > 10:
                f.write("The service migration approach significantly improved load balancing, ")
                f.write(f"reducing load variance by {comparative_metrics.get('load_balance_improvement', 0):.1f}% on average. ")
            else:
                f.write("The service migration approach had a limited impact on load balancing. ")
                
            f.write("\n\nFor detailed analysis, refer to the visualization plots in the analysis directory.")
            
        print(f"Summary report created at {analysis_dir}/summary_report.md")
        return True
    
    def analyze_all(self):
        """Run all analysis methods"""
        if self.load_data():
            self.generate_advanced_plots()
            self.calculate_comparative_metrics()
            self.create_summary_report()
            print("Analysis completed successfully")
            return True
        else:
            print("Analysis failed due to data loading issues")
            return False

def main():
    """Main function to run the analysis"""
    analyzer = SimulationAnalyzer()
    analyzer.analyze_all()

if __name__ == "__main__":
    main()