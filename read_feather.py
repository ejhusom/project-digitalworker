#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
import pandas as pd

dfs = []

for i in range(1,7):
    print(i)
    df = pd.read_feather(f"batch_dataset_{i}.feather")
    df = df.iloc[:,:17]
    dfs.append(df)

combined = pd.concat(dfs)

print(combined)
combined.to_csv("digitalworker-combined.csv")

