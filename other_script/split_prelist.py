import pandas as pd
import os

SOURCE_FILE = 'source_matcha_pro_all/combined_data_prelist.csv'
OUT_FULL = 'result/split_prelist/full'
OUT_COMPLETE = 'result/split_prelist/complete'
OUT_INCOMPLETE = 'result/split_prelist/incomplete'
CHUNK_SIZE = 50_000
REQUIRED_COLS = ['kbli', 'kategori', 'nomor_hp', 'email']

for path in [OUT_FULL, OUT_COMPLETE, OUT_INCOMPLETE]:
    os.makedirs(path, exist_ok=True)

# Track which output files already have a header written
headers_written = {'full': set(), 'complete': set(), 'incomplete': set()}

total = complete = incomplete = 0

for chunk in pd.read_csv(SOURCE_FILE, dtype=str, chunksize=CHUNK_SIZE):
    chunk['_kode4'] = chunk['kode_wilayah'].str[:4]

    is_complete = chunk[REQUIRED_COLS].apply(
        lambda col: col.notna() & (col.str.strip() != '')
    ).all(axis=1)

    for kode, group in chunk.groupby('_kode4'):
        rows = group.drop(columns=['_kode4'])

        # Full
        dest = f'{OUT_FULL}/{kode}_full.csv'
        write_header = kode not in headers_written['full']
        rows.to_csv(dest, mode='a', index=False, header=write_header)
        headers_written['full'].add(kode)

        # Complete / incomplete
        c_rows = rows[is_complete[group.index]]
        i_rows = rows[~is_complete[group.index]]

        if not c_rows.empty:
            dest = f'{OUT_COMPLETE}/{kode}_complete.csv'
            write_header = kode not in headers_written['complete']
            c_rows.to_csv(dest, mode='a', index=False, header=write_header)
            headers_written['complete'].add(kode)

        if not i_rows.empty:
            dest = f'{OUT_INCOMPLETE}/{kode}_incomplete.csv'
            write_header = kode not in headers_written['incomplete']
            i_rows.to_csv(dest, mode='a', index=False, header=write_header)
            headers_written['incomplete'].add(kode)

    total += len(chunk)
    complete += is_complete.sum()
    incomplete += (~is_complete).sum()
    print(f"  processed {total:,} rows...", end='\r')

print(f"\nTotal rows     : {total:,}")
print(f"Complete rows  : {complete:,}")
print(f"Incomplete rows: {incomplete:,}")
print("Done.")
