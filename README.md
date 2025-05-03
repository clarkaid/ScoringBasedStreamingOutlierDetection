# Scoring-Based Streaming Algorithms for Outlier Detection

This repository contains implementations of four popular streaming algorithms for outlier detection, along with testing code comparing their performances. We examine:
1. Incremental Local Outlier Factor (ILOF), proposed in [this paper](https://ieeexplore.ieee.org/document/4221341). Our implementation is in the file `IncrementalLOF.py`
2. Memory Efficient Incremental Local Outlier Factor (MILOF), proposed in [this paper](https://ieeexplore.ieee.org/document/7530918). Our implementation is in the file `milof.py`
3. Density Summarizing Incremental Local Outlier Factor (DILOF), proposed in [this paper](https://doi.org/10.1145/3219819.3220022). Our implementation is in the file `dilof.py`.
4. Kernel Density Estimation with Reservoirs (KDEWR), proposed in [this paper](https://doi.org/10.3390/s20051261). Our implementation is in the file `kdwer.py`.

We include code testing these algorithms on simulated and real world datasets in the following files:
* `testing_network_traffic.py`
* `testing_credit_card_fraud.py`
* `simulatedTesting.ipynb`

A list of packages needed to work with our implementations is in `requirements.txt`. 
