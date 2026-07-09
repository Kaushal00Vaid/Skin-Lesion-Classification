import pandas as  pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("../data/HAM10000_metadata.csv")

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=876567
)

train_idx, val_idx = next(sgkf.split(df, df['dx'], df['lesion_id']))
df['split'] = 'train'
df.loc[val_idx, 'split'] = 'val'

output_filename = '../data/HAM10000_metadata_split_80_20.csv'
df.to_csv(output_filename, index=False)
print(f"\nSuccessfully saved fixed split to: {output_filename}")