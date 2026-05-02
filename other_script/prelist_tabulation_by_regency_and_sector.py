import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.parent

df = pd.read_csv(base_dir / 'source_matcha_pro_all' / 'combined_data_prelist.csv', dtype=str)

df['kode_kabupaten'] = df['kode_wilayah'].str[:4]
df['kategori'] = df['kategori'].fillna('blank')

tabulation = (
    df.groupby(['kode_kabupaten', 'skala_usaha', 'kategori'])
    .size()
    .unstack(fill_value=0)
)

tabulation.columns.name = None
tabulation = tabulation.reset_index()

print(tabulation)
tabulation.to_csv(base_dir / 'result' / 'prelist_tabulation_by_regency_and_sector.csv', index=False)
print("Saved to result/prelist_tabulation_by_regency_and_sector.csv")
