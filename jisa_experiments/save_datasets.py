import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration 
INPUT_FOLDER = "geant-datasets"
AGGREGATED_FOLDER = "aggregated_6h_datasets"
REPORT_FILE = "aggregation_6h_report.txt"
TIMESTAMP_COL = "Data"
VALUE_COL = "Vazao"

def setup_folders():
    """Create necessary folders if they don't exist"""
    Path(AGGREGATED_FOLDER).mkdir(exist_ok=True)

def aggregate_by_6h(df, time_column=TIMESTAMP_COL, value_column=VALUE_COL):
    """
    Aggregate time series values in fixed 6-hour blocks.
    Keeps NaN for intervals without measurements.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.set_index(time_column).sort_index()
    
    df_resampled = df[value_column].resample("6h").mean()
    df_resampled = df_resampled.asfreq("6h")
    
    return df_resampled

def process_file(file_path, report_handle):
    """Process a single file and aggregate to 6 hours"""
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    
    report_handle.write(f"\n{'='*80}\n")
    report_handle.write(f"PROCESSING FILE: {filename}\n")
    report_handle.write(f"{'='*80}\n")
    report_handle.write(f"Original dataset shape: {df.shape}\n")
    report_handle.write(f"Date range: {df[TIMESTAMP_COL].min()} to {df[TIMESTAMP_COL].max()}\n\n")
    
    # Aggregate to 6 hours
    series_6h = aggregate_by_6h(df, TIMESTAMP_COL, VALUE_COL)
    missing_6h = series_6h.isna().sum()
    
    report_handle.write(f"6-hour aggregation results:\n")
    report_handle.write(f"  Total points: {len(series_6h)}\n")
    report_handle.write(f"  Missing values: {missing_6h} ({missing_6h/len(series_6h)*100:.2f}%)\n")
    report_handle.write(f"  Valid values: {len(series_6h) - missing_6h}\n")
    report_handle.write(f"  First 10 values:\n{series_6h.head(10)}\n")
    report_handle.write(f"  Last 10 values:\n{series_6h.tail(10)}\n\n")
    
    # Save aggregated data
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(AGGREGATED_FOLDER, f"{base_name}_6h.csv")
    series_6h.to_csv(output_path, header=True)
    
    report_handle.write(f"Saved to: {output_path}\n")
    
    return series_6h

def process_all_files():
    """Main function to process all CSV files in the input folder"""
    setup_folders()
    
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as report:
        report.write("6-HOUR AGGREGATION REPORT\n")
        report.write("="*80 + "\n")
        report.write(f"Input folder: {INPUT_FOLDER}\n")
        report.write(f"Output folder: {AGGREGATED_FOLDER}\n")
        report.write(f"Aggregation interval: 6 hours\n")
        report.write("="*80 + "\n")
        
        processed = 0
        failed = 0
        
        for csv_file in csv_files:
            file_path = os.path.join(INPUT_FOLDER, csv_file)
            
            try:
                process_file(file_path, report)
                processed += 1
                print(f"✓ Processed: {csv_file}")
                
            except Exception as e:
                failed += 1
                error_msg = f"ERROR processing {csv_file}: {str(e)}\n\n"
                report.write(error_msg)
                print(f"✗ Failed: {csv_file} - {str(e)}")
                continue
        
        # Summary
        report.write(f"\n{'='*80}\n")
        report.write(f"SUMMARY\n")
        report.write(f"{'='*80}\n")
        report.write(f"Total files found: {len(csv_files)}\n")
        report.write(f"Successfully processed: {processed}\n")
        report.write(f"Failed: {failed}\n")
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"{'='*80}")
    print(f"Total files: {len(csv_files)}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"Aggregated datasets saved to: {AGGREGATED_FOLDER}/")

if __name__ == "__main__":
    process_all_files()