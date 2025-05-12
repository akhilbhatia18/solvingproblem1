import pandas as pd
import os
import glob

def multiple_excel_to_single_csv(folder_path, output_csv, sheet_names=None, include_filename=True, include_sheet_name=True):
    """
    Convert multiple sheets from multiple Excel files in a folder to a single CSV file.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing Excel files
    output_csv : str
        Path to the output CSV file
    sheet_names : list or None, optional
        List of sheet names to include. If None, all sheets will be included.
    include_filename : bool, optional
        Whether to include the source filename as a column in the output
    include_sheet_name : bool, optional
        Whether to include the source sheet name as a column in the output
    
    Returns:
    --------
    str
        Path to the created CSV file
    """
    # Get all Excel files in the folder
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx")) + glob.glob(os.path.join(folder_path, "*.xls"))
    # Sort files alphabetically
    excel_files.sort()

    if not excel_files:
        print(f"No Excel files found in {folder_path}")
        return None
    
    # Create a list to store all dataframes
    all_dfs = []
    
    # Process each Excel file
    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        try:
            # Get all sheet names if not specified
            if sheet_names is None:
                xl = pd.ExcelFile(file_path)
                sheets_to_process = xl.sheet_names
            else:
                sheets_to_process = sheet_names
            
            # Process each sheet
            for sheet in sheets_to_process:
                try:
                    # Read the sheet
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    
                    # Add filename and sheet name as columns if requested
                    if include_filename:
                        df['Source_File'] = file_name
                    if include_sheet_name:
                        df['Source_Sheet'] = sheet
                    
                    # Add to the list of dataframes
                    all_dfs.append(df)
                    print(f"  - Added sheet '{sheet}' with {len(df)} rows")
                except Exception as e:
                    print(f"  - Error processing sheet '{sheet}': {e}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    if not all_dfs:
        print("No data was extracted from the Excel files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Write to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"\nCombined {len(all_dfs)} sheets into {output_csv}")
    print(f"Total rows: {len(combined_df)}")
    
    return output_csv

if __name__ == "__main__":

    # Convert Excel files to a single CSV
    multiple_excel_to_single_csv(
        "data/DataSet",
        "data/DataSet/telecom_dataset_output.csv"
    )