import os
import pandas as pd

def combine_csv_files(folder_path, output_file, max_files=9):
    # List to hold dataframes
    dataframes = []
    
    # Iterate through all files in the folder
    for i, file_name in enumerate(os.listdir(folder_path)):
        if file_name.endswith('.csv'):
            if i >= max_files:  # Stop after processing max_files
                break
            file_path = os.path.join(folder_path, file_name)
            # Read CSV file and append to the list
            dataframes.append(pd.read_csv(file_path))
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

# Example usage
if __name__ == "__main__":
    folder_path = "results"  # Replace with your folder path
    output_file = "results/combined_output.csv"  # Replace with your desired output file name
    combine_csv_files(folder_path, output_file)