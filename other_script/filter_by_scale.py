import pandas as pd
import os
import sys

# Set the target scale value here
TARGET_SCALE_VALUE = "UB"

def main():
    if not TARGET_SCALE_VALUE:
        print("Please set the TARGET_SCALE_VALUE variable in the script.")
        sys.exit(1)

    # Resolve paths relative to the project root (assuming script is in other_script/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(project_root, "source_matcha_pro_all", "combined_data.csv")
    output_dir = os.path.join(project_root, "result")
    
    # Replace spaces or special characters in target value for filename safety
    safe_value = str(TARGET_SCALE_VALUE).replace('/', '_').replace('\\', '_')
    output_file = os.path.join(output_dir, f"filter_by_scale_{safe_value}.csv")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Reading {input_file} (this might take a moment)...")
    try:
        # Using dtype=str to avoid DtypeWarnings on mixed columns
        df = pd.read_csv(input_file, dtype=str)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    if 'skala_usaha' not in df.columns:
        print(f"Error: 'skala_usaha' column not found in the CSV. Available columns:")
        print(", ".join(df.columns))
        sys.exit(1)

    print(f"Filtering where 'skala_usaha' == '{TARGET_SCALE_VALUE}'...")
    filtered_df = df[df['skala_usaha'] == TARGET_SCALE_VALUE]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(filtered_df)} rows to {output_file}...")
    filtered_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
