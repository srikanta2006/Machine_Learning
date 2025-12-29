import seaborn as sns
import numpy as np
import pandas as pd


def main():
    try:
        df = sns.load_dataset("taxis")
    except Exception as e:
        print("Failed to load 'taxis' dataset:", e)
        return

    print("head:-")
    print(df.head())

    print("columns:-")
    for col in df.columns:
        print(col)

    print("shape:-", df.shape)

    print("describe:-")
    # include='all' to show non-numeric summaries too
    print(df.describe(include='all'))


if __name__ == '__main__':
    main()
