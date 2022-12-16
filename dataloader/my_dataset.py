import pandas as pd 
import os 
from sklearn.model_selection import train_test_split

def load_csv_file(path = './data', val_split = 0.2):
    df_train = pd.read_csv(os.path.join(path, 'train/label.csv'))
    train, val = train_test_split(df_train, test_size=val_split, random_state=42)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train['split'] = 'train'
    val['split'] = 'val'

    test = pd.DataFrame()
    test['fname'] = os.listdir(os.path.join(path, 'public/videos'))
    test['liveness_score'], test['split'] = None, 'test'
    df: pd.DataFrame = pd.concat((train, val, test), ignore_index=True)

    return df
