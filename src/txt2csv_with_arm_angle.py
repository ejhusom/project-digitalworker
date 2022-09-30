#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Converting DigitalWorker txt-files to csv-files.

Author:
    Erik Johannes Husom

Created:
    2022-09-28

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_files(dir_path, file_extension=[]):
    """Find files in directory.

    Args:
        dir_path (str): Path to directory containing files.
        file_extension (str): Only find files with a certain extension. Default
            is an empty string, which means it will find all files.

    Returns:
        filepaths (list): All files found.

    """

    filepaths = []

    if type(file_extension) is not list:
        file_extension = [file_extension]

    for extension in file_extension:
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(extension):
                filepaths.append(dir_path + "/" + f)

    return filepaths

def txt2csv(filenames):

    headers = ["Timestamp","Class","Trunk_AccX","Trunk_AccY","Trunk_AccZ","Arm_AccX","Arm_AccY","Arm_AccZ","Hip_AccX","Hip_AccY","Hip_AccZ","Thigh_AccX","Thigh_AccY","Thigh_AccZ","Calf_AccX","Calf_AccY","Calf_AccZ","Trunk_Inclination","Trunk_Forward","Trunk_Sideways","Arm_Inclination","Arm_Forward","Arm_Sideways","Hip_Inclination","Hip_Forward","Hip_Sideways","Thigh_Inclination","Thigh_Forward","Thigh_Sideways","Calf_Inclination","Calf_Forward","Calf_Sideways","Arm_Elevation"]

    for f in filenames:
            print(f)
            df = pd.read_csv(f, names=headers)
            # df["Timestamp"] = pd.to_datetime(df["Timestamp"]).astype(int) / 10**9
            # df["Timestamp"] = df["Timestamp"].astype(np.int64)
            df = df.iloc[:,:21]
            df["Class"] = df["Class"].astype(np.int64)
            df["Arm_Inclination"] = df["Arm_Inclination"].astype(np.float64)
            del df["Trunk_Inclination"]
            del df["Trunk_Forward"]
            del df["Trunk_Sideways"]
            print(df.info())
            print(df.describe())

            df.to_csv(os.path.basename(f) + ".csv", index=False)

if __name__ == '__main__':

    txt2csv(sys.argv[1:])
