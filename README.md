# accelstm

This repository contains the source code associated with the SciPy 2018 Proceedings paper and associated poster "Developing a Start-to-Finish Pipeline for Accelerometer-Based Activity Recognition Using Long Short-Term Memory Recurrent Neural Networks.”

Accelerometer data is read in and formatted for a Data Analysis Pipeline within the folder src/data/.
In src/models/, a baseline LSTM is optimized based on a wide range of hyperparameter settings found in literature (i.e., a thirty-study meta-analysis style overview of the field of human activity recognition (HAR) using LSTM models). The optimized LSTM is then incorporated in a proposed Data Analysis Pipeline intended to foster reproducibility and scientific rigor within the field. 
