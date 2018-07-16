# accelstm
## Developing a Start-to-Finish Pipeline for Accelerometer-Based Activity Recognition Using Long Short-Term Memory Recurrent Neural Networks

This repository contains the source code associated with the SciPy 2018 Proceedings paper and associated poster "Developing a Start-to-Finish Pipeline for Accelerometer-Based Activity Recognition Using Long Short-Term Memory Recurrent Neural Networks.”

**Accelerometer data** is read in and formatted for a Data Analysis Pipeline within the folder `src/data/`.
In `src/models/`, a **baseline LSTM is optimized** based on a wide range of hyperparameter settings found throughout literature (i.e., a 30-study meta-analysis style overview of the field of human activity recognition (HAR) using LSTM models). 
The optimized LSTM is **then incorporated in a proposed Data Analysis Pipeline** intended to foster reproducibility and scientific rigor within the field.
The poster and other documents are found in the folder `docs/`. 

As of July 2018, this repo has been made to work on the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) from Reyes-Ortiz, et. al. (2013). Solely the triaxial *total* accelerometer signal (vs. the body-only signal with gravity component removed) as well as the [UCI HMP Dataset] (https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer) from Bruno, et. al. (2012). 

**Note:** If you would like help formatting another dataset for this pipeline, or if you have another issues/comments, please make an Issue and include an `@xtianmcd` tag.
Additionally, if you use this code for any project, please let me know at clm121@uga.edu and cite the repo, and feel free to submit a PR to upload any additional code you might write to the repo. 

