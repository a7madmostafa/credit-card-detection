import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_STATE

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f'Data Loaded with {len(df)} rows')
    df = df.drop_duplicates()
    print(f'Data after drop duplicates: {len(df)} rows')
    return df


def split_data(df):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print('Data Splitted:')
    print(f'X_train: {len(X_train)} rows')
    print(f'X_test: {len(X_test)} rows')
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = load_data()
    split_data(df)