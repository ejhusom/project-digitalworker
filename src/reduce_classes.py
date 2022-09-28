import os
import sys

import numpy as np
import pandas as pd


def reduce_classes(filenames):

    for filename in filenames:

        df = pd.read_csv(filename)

        # Remove "Sensor off"
        df = df[df.Class != 0]
        # Remove "Rowing"
        df = df[df.Class != 10]
        # Remove "Cycling"
        df = df[df.Class != 9]

        # Combine walk fast and slow
        df['Class'] = df['Class'].replace(to_replace=11, value=6)
        df['Class'] = df['Class'].replace(to_replace=12, value=6)

        #print(os.path.basename(filename) + "_reduced_classes.csv")
        df.to_csv(os.path.basename(filename) + "_reduced_classes.csv", index=False)
        

        # if (9 in df.Class.unique()
        #         or 9 in df.Class.unique()
        #         or 0 in df.Class.unique()
        #         or 10 in df.Class.unique()
        #         or 11 in df.Class.unique()
        #         or 12 in df.Class.unique()
        #         ):
        #     print(df.Class.unique())

if __name__ == '__main__':

    filenames = sys.argv[1:]

    reduce_classes(filenames)
