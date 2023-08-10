import pandas as pd
from pathlib import Path


dir = ''
mode = 'train'
names = [
    '',
    '',
]
output_dir=''


df_paths = []
for name in names:
    df_paths.append(f'{dir}/{name}/{mode}.csv')

print(df_paths)


print('Combining data ...')

prev_max = 0
dfs = []
for df_path in df_paths:

    df = pd.read_csv(df_path)
    df['class'] += prev_max
    prev_max = df['class'].max() + 1

    dfs.append(df)

df = pd.concat(dfs)

print(f'Saving data to {output_dir} ...')

path = Path(output_dir) / f'{mode}.csv'

df.to_csv(path, index=None)
