#!/bin/bash
# ===================================================================
# File:     add_header_to_original_files.sh
# Author:   Erik Johannes Husom
# Created:
# -------------------------------------------------------------------
# Description: For the DigitalWorker data, add column names to files.
# ===================================================================

header="Timestamp,Class,Trunk_AccX,Trunk_AccY,Trunk_AccZ,Arm_AccX,Arm_AccY,Arm_AccZ,Hip_AccX,Hip_AccY,Hip_AccZ,Thigh_AccX,Thigh_AccY,Thigh_AccZ,Calf_AccX,Calf_AccY,Calf_AccZ,Trunk_Inclination,Trunk_Forward,Trunk_Sideways,Arm_Inclination,Arm_Forward,Arm_Sideways,Hip_Inclination,Hip_Forward,Hip_Sideways,Thigh_Inclination,Thigh_Forward,Thigh_Sideways,Calf_Inclination,Calf_Forward,Calf_Sideways,ID" # ,Arm_Elevation,ID"

for file in "$@"; do
    echo $file
    echo $header > "outfile.csv"
    cat $file >> "outfile.csv"
    cat "outfile.csv" > "new_${file}.csv"
done


