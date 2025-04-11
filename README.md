# Methods for applying Mapper to EEG data
This code is for the paper titled, "MapperEEG: A Topological Approach to Brain State Clustering in EEG Recordings"

## Data:
Zhibin Zhou is the data collector and manager and can be reached on [LinkedIn](https://www.linkedin.com/in/zhibin-zhou-b382349a/). 
The dataset consists of 12 files of tapping sessions for six pairs of tappers. For additional explanation, please see paper.
Packages necessary: os, numpy, pandas, pickle, networkx, matplotlib, scipy, h5py, collections, sklearn, warnings, umap, random, datetime, kmapper

## Scripts:
Each of the scripts are described briefly below:

-apply_mapper_concat_synco_synchro.py -- Applies the M-EEG algorithm to each tapper's concatenated synchronized and syncopated sessions and calculates the Qmod score.
-comparisons_*.py -- Calculates the five different errors (f1 min, f1 max, f1 weighted, silhouette score, and Davies Bouldin) for each of the four clustering algorithms (DBSCAN, k-Nearest Neighbors, Gaussian Mean Mixture, and MEEG), respectively.
-functions.py -- The file containing all functions used in the scripts.
-band_search.py -- Calculates the Mapper graph outputs across the five different bands for all 12 tappers.
