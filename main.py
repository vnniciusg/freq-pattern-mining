import os
import warnings

import pandas as pd
from kagglehub import dataset_download

warnings.filterwarnings("ignore")

def load_df(handle: str = "rukenmissonnier/real-market-data") -> pd.DataFrame:

    path = dataset_download(handle)
    files = os.listdir(path)

    for file in files:
        if file.endswith(".csv"):
            return pd.read_csv(os.path.join(path, file), sep=";").astype(bool)

def main():
    df = load_df()
    print(df.head())


if __name__ == "__main__":
    main()
