import pandas as pd
from pathlib import Path


def prepare_cnn_dataset(annotationsdf_path, output_dir, n_classes, test_size=0.2):

    df = pd.read_csv(annotationsdf_path)
    
    # Exclude unselected classes
    df = df[(df['class'] >= 0) & (df['class'] < n_classes)]

    df_test = df.groupby('class', group_keys=False).apply(lambda x: x.sample(int(len(x)*test_size), replace=False))
    df_train = df[~df.index.isin(df_test.index)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / 'train.csv', index=False)
    df_test.to_csv(output_dir / 'test.csv', index=False)

