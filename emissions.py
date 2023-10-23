#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-liner describing module.

Author:
    Erik Johannes Husom

Created:
    2021

"""
import numpy as np
import pandas as pd

df = pd.read_csv("emissions.csv")
df["duration_min"] = df["duration"] / 60

stages = df["project_name"].unique()
summation = df.sum()

groups_sum = df.groupby(["project_name"]).sum()
groups_avg = df.groupby(["project_name"]).mean()
groups_avg_sum = groups_avg.sum()
groups_std = df.groupby(["project_name"]).std()
groups_var = df.groupby(["project_name"]).var()
groups_var_sum = groups_var.sum()
groups_var_sum_sqrt = np.sqrt(groups_var_sum)

print(groups_avg["energy_consumed"])
print(groups_avg)
print(groups_avg_sum)
print(groups_std)
print(groups_std["energy_consumed"])
# print(groups_var)
# print(groups_var_sum)
print(groups_var_sum_sqrt)
# print(stages)
# print(summation)





