import os
import pandas as pd
import re


def extract_kategori(value):
    match = re.search(r'Kategori: ([A-Z])', str(value))
    if match:
        return match.group(1)
    return 'G'

# Coordinate validation and normalization
def normalize_coordinate(val):
    if pd.isnull(val):
        return None
    # 1. Trim spaces
    val = str(val).strip()
    # 2. Replace comma decimal separator with point
    val = val.replace(',', '.')
    # 3. Repair malformed decimal placement (e.g., '1123456' -> '112.3456' if needed)
    # Try to convert directly
    try:
        float(val)
        return val
    except ValueError:
        pass
    # Try to repair: if only digits and more than 5 chars, insert decimal after 2 or 3 digits
    if val.replace('.', '').isdigit():
        if len(val) > 5 and '.' not in val:
            # Try after 2 or 3 digits
            for i in [2, 3]:
                try:
                    fixed = val[:i] + '.' + val[i:]
                    float(fixed)
                    return fixed
                except ValueError:
                    continue
    return None


def validate_and_normalize_coordinates(lat, lon):
    lat_norm = normalize_coordinate(lat)
    lon_norm = normalize_coordinate(lon)
    try:
        lat_f = float(lat_norm)
        lon_f = float(lon_norm)
        # Latitude valid range: -90 to 90, Longitude: -180 to 180
        if -90 <= lat_f <= 90 and -180 <= lon_f <= 180:
            return ('valid', lat_norm, lon_norm)
        else:
            return ('invalid', None, None)
    except (TypeError, ValueError):
        return ('invalid', None, None)

source_dir = 'source_matcha_pro_all'
split_result = True  # Set to True to split result files, False for single file
chunksize = 10000  # Adjust chunk size as needed
debug_mode = True  # Set to True to enable debug output

split_dir = 'result/extract_kbli/split'
output_path = 'result/extract_kbli/matchapro_extracted_category.csv'
os.makedirs(split_dir, exist_ok=True)

all_chunks = []
for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_dir, filename)
        if debug_mode:
            print(f"Processing file: {filename}")
        chunk_idx = 0
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            if 'kegiatan_usaha' in chunk.columns:
                chunk['extracted_kategori'] = chunk['kegiatan_usaha'].apply(extract_kategori)
            else:
                chunk['extracted_kategori'] = 'G'


            # Coordinate validation and normalized columns
            if 'latitude' in chunk.columns and 'longitude' in chunk.columns:
                def coord_row(row):
                    status, lat_valid, lon_valid = validate_and_normalize_coordinates(row['latitude'], row['longitude'])
                    return pd.Series({
                        'extracted_coordinate_validation': status,
                        'valid_latitude': lat_valid,
                        'valid_longitude': lon_valid
                    })
                coord_result = chunk.apply(coord_row, axis=1)
                chunk['extracted_coordinate_validation'] = coord_result['extracted_coordinate_validation']
                chunk['valid_latitude'] = coord_result['valid_latitude']
                chunk['valid_longitude'] = coord_result['valid_longitude']
            else:
                chunk['extracted_coordinate_validation'] = 'invalid'
                chunk['valid_latitude'] = None
                chunk['valid_longitude'] = None

            if split_result:
                split_filename = f"{os.path.splitext(filename)[0]}_chunk{chunk_idx}.csv"
                split_path = os.path.join(split_dir, split_filename)
                chunk.to_csv(split_path, index=False)
                if debug_mode:
                    print(f"  Saved chunk {chunk_idx} to {split_path} (rows: {len(chunk)})")
            else:
                all_chunks.append(chunk)
            if debug_mode and not split_result:
                print(f"  Processed chunk {chunk_idx} (rows: {len(chunk)})")
            chunk_idx += 1

if not split_result and all_chunks:
    combined_df = pd.concat(all_chunks, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved extracted categories to {output_path}')
elif not all_chunks:
    print('No CSV files found in source_matcha_pro_all')
