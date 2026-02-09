import pandas as pd
import os
from pathlib import Path

def process_vlookup_with_chunks():
    """
    Process all Excel/CSV files in gc_transform/source folder.
    Perform VLOOKUP against large combined_data.csv using chunked reading.
    Output results for records where gc_result is empty.
    Optimized to scan combined_data.csv only ONCE for all source files.
    """
    
    # Define paths
    source_folder = Path("gc_transform/source")
    result_folder = Path("gc_transform/result")
    combined_data_path = Path("source_matcha_pro_all/combined_data.csv")
    
    # Create result folder if it doesn't exist
    result_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all Excel and CSV files from source folder
    source_files = list(source_folder.glob("*.xlsx")) + list(source_folder.glob("*.csv"))
    
    if not source_files:
        print("No Excel or CSV files found in gc_transform/source")
        return
    
    print(f"Found {len(source_files)} file(s) to process")
    
    # Step 1: Load all source files and collect ALL unique idsbr values
    print("\n" + "=" * 70)
    print("STEP 1: Loading all source files")
    print("=" * 70)
    
    source_data = {}  # Store each source file's data
    all_idsbr = set()  # Collect all unique idsbr values across all files
    
    for source_file in source_files:
        print(f"\nLoading: {source_file.name}")
        
        # Read source file
        if source_file.suffix == '.xlsx':
            df_source = pd.read_excel(source_file)
        else:
            df_source = pd.read_csv(source_file)
        
        print(f"  - Loaded {len(df_source)} rows")
        
        # Check if idsbr column exists
        if 'idsbr' not in df_source.columns:
            print(f"  - ERROR: 'idsbr' column not found")
            print(f"  - Available columns: {', '.join(df_source.columns)}")
            continue
        
        # Store source data
        source_data[source_file] = df_source
        
        # Collect unique idsbr values
        idsbr_values = df_source['idsbr'].dropna().unique()
        all_idsbr.update(idsbr_values)
        print(f"  - Found {len(idsbr_values)} unique idsbr values")
    
    if not all_idsbr:
        print("\nNo idsbr values found to lookup!")
        return
    
    all_idsbr_list = list(all_idsbr)
    print(f"\nTotal unique idsbr values across all files: {len(all_idsbr_list)}")
    
    # Step 2: Scan combined_data.csv ONCE and collect all matching records
    print("\n" + "=" * 70)
    print("STEP 2: Scanning combined_data.csv (ONE TIME ONLY)")
    print("=" * 70)
    
    chunk_size = 100000  # Process 100k rows at a time
    matched_data = []
    
    for chunk_num, chunk in enumerate(pd.read_csv(combined_data_path, chunksize=chunk_size, low_memory=False), 1):
        # Filter chunk for matching idsbr values
        matched_chunk = chunk[chunk['idsbr'].isin(all_idsbr_list)]
        
        if len(matched_chunk) > 0:
            matched_data.append(matched_chunk)
        
        if chunk_num % 10 == 0:
            print(f"  Processed {chunk_num * chunk_size:,} rows...", end='\r')
    
    print(f"\n  Completed scanning combined_data.csv")
    
    # Combine all matched chunks
    if matched_data:
        df_lookup = pd.concat(matched_data, ignore_index=True)
        print(f"  Found {len(df_lookup)} matching records total")
    else:
        print(f"  No matching records found")
        df_lookup = pd.DataFrame(columns=['idsbr', 'perusahaan_id', 'latitude', 'longitude', 'gcs_result'])
    
    # Step 3: Process each source file with the lookup data
    print("\n" + "=" * 70)
    print("STEP 3: Creating result files")
    print("=" * 70)
    
    for source_file, df_source in source_data.items():
        print(f"\nProcessing: {source_file.name}")
        
        # Perform VLOOKUP (merge) for this specific source file
        df_result = df_source[['idsbr']].merge(
            df_lookup[['idsbr', 'perusahaan_id', 'latitude', 'longitude', 'gcs_result']],
            on='idsbr',
            how='left'
        )
        
        # Filter only records where gcs_result is empty/null
        df_result = df_result[df_result['gcs_result'].isna() | (df_result['gcs_result'] == '')]
        
        # Extract value from filename for hasilgc (between -- and --)
        import re
        filename = source_file.stem
        
        # Try to find text between -- and --
        match = re.search(r'--(.+?)--', filename)
        hasilgc_value = match.group(1).strip() if match else ''
        
        print(f"  - Extracted hasilgc value: '{hasilgc_value}'")
        
        # Select and rename columns for output
        df_output = df_result[['idsbr', 'perusahaan_id', 'latitude', 'longitude']].copy()
        df_output['hasilgc'] = hasilgc_value  # Fill with extracted value
        
        # Remove duplicates based on idsbr
        df_output = df_output.drop_duplicates(subset=['idsbr'])
        
        print(f"  - Filtered to {len(df_output)} records with empty gc_result")
        
        # Generate output filename
        output_filename = source_file.stem + "_result.csv"
        output_path = result_folder / output_filename
        
        # Write to CSV
        df_output.to_csv(output_path, index=False)
        print(f"  - Saved result to: {output_path}")
        print(f"  - Output columns: {', '.join(df_output.columns)}")

if __name__ == "__main__":
    print("=" * 70)
    print("GC Transform - VLOOKUP Processing Script")
    print("=" * 70)
    process_vlookup_with_chunks()
    print("\n" + "=" * 70)
    print("Processing completed!")
    print("=" * 70)
