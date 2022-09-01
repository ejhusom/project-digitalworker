#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
import os

import h5py
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

def txt2hdf5():

    headers = ["Timestamp","Class","Trunk_AccX","Trunk_AccY","Trunk_AccZ","Arm_AccX","Arm_AccY","Arm_AccZ","Hip_AccX","Hip_AccY","Hip_AccZ","Thigh_AccX","Thigh_AccY","Thigh_AccZ","Calf_AccX","Calf_AccY","Calf_AccZ","Trunk_Inclination","Trunk_Forward","Trunk_Sideways","Arm_Inclination","Arm_Forward","Arm_Sideways","Hip_Inclination","Hip_Forward","Hip_Sideways","Thigh_Inclination","Thigh_Forward","Thigh_Sideways","Calf_Inclination","Calf_Forward","Calf_Sideways","Arm_Elevation"]
    txt_files = find_files(".", file_extension=["txt"])

    for f in txt_files:
        df = pd.read_csv(f, names=headers)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).astype(int) / 10**9
        # df["Timestamp"] = df["Timestamp"].astype(np.int64)
        df = df.iloc[:,:17]
        print(df.info())

        df.to_csv(f"{f}.csv")

        # df = df.to_numpy()
        # np.savetxt(f"{f}.npy", df)
        # df.to_hdf(f"{f}.hdf5", key="df")
        break

if __name__ == '__main__':

    txt2hdf5()
