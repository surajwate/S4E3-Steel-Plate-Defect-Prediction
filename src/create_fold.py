import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def create_fold(fold):
    # import data
    data = pd.read_csv('./input/train.csv')
    print(f"Data shape: {data.shape}")

    data['kfold'] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    # Define the target columns for multi-label classification
    target_columns = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # Create a single target for statification by checking if any label is present
    data["faulty"] = data[target_columns].max(axis=1)

    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    for f, (t_, v_) in enumerate(skf.split(X=data, y=data["faulty"].values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop("faulty", axis=1)

    data.to_csv('./input/train_folds.csv', index=False)
    print(f"Created {fold} folds.")

if __name__ == '__main__':
    create_fold(5)