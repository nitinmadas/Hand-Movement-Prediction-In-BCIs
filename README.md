# Hand-Movement-Prediction-In-BCIs

This repository provides the implemention code for the Brain Computer Interface Channel Whitening [(BCICW)](https://www.mdpi.com/1424-8220/22/16/6042) proposed in the paper:

Choi, H.; Park, J.; Yang, Y.-M. Whitening Technique Based on Gram–Schmidt Orthogonalization for Motor Imagery Classification of Brain-Computer Interface Applications. Sensors 2022, 22, 6042. https://doi.org/10.3390/s22166042

#

## Problem
In BCI research, major challenges are
- Accurate classification of different mental activities (Motor Imagery)
- Accuracy Consistency across subjects by addressing the variance in brain signals

## Solution:
- Apply Eigen Face Analysis (EFA) and Linear Discriminant Analysis (LDA) for mental activity( Motor Imagery) classification
- Apply Brain-Computer Interface Channel Whitening (BCICW) for signal processing before feature engineering and classification

## Dataset:
Epoched Electroencephalogram (EEG) signal data for channels C3, CZ, and C4 of 9 subjects ([link](https://www.kaggle.com/competitions/ucsd-neural-data-challenge))


## Workflow chart
<p align="center">
Basic Workflow
</p>
<p align="center">
<img src="https://nitinmadas.github.io/assets/img/projects/bci_project.png" alt="The components of the proposed BCICW model" width="400"/>
</p>

<p align="center">
The components of the BCICW model
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/KJ-999/Controlling-Machines-with-Human-Brain/main/Assets/Workflow_BCI.webp?token=GHSAT0AAAAAACOLUMH7RMFHSR7Q5HDTYFDWZOSJHQA" alt="The components of the proposed BCICW model" width="400"/>
</p>


## About [BCICW](https://www.mdpi.com/1424-8220/22/16/6042) : 
* Whitening Technique [(BCICW)](https://www.mdpi.com/1424-8220/22/16/6042) is a preprocessing step where we use Gram–Schmidt Orthogonalization to de-correlate the mixed signal data i.e. to minimize the variance in accuracy among subjects. 
* [Whitening Transform](https://www.mdpi.com/1424-8220/22/16/6042) is aimed to provide a unit variance and a minimum covariance for the given random data due to which it minimizes the dependency between experimental participants or subjects which is a essential key factor to solve classification problems.

<p align="center">
<img src="https://github.com/KJ-999/Controlling-Machines-with-Human-Brain/blob/main/Assets/Whitening.png" alt="Covariance matrix for Whitening Transform" width="700"/>
</p>

* We tried to minimize channel dependence among the measured data in electrodes by maximizing the diagonal terms to unity and minimizing the off-diagonal terms, i.e., whitening the data.
* Because of whitening in the channel direction, the independent eigenface for each class is unique and distinguishable. In addition, the Euclidean distance between the coefficients of left and right classes has been increased. Those contributions result in improved accuracy and a reduced variance.

## Code
The [Automated_BCICW_EFA_LDA.py](https://github.com/KJ-999/Controlling-Machines-with-Human-Brain/blob/main/Automated_BCICW_EFA_LDA.ipynb) loads the data using pickle and apply two approaches : 
1. In first approach we applied **EFA** & **LDA** **without** applying the [Whitening Technique](https://www.mdpi.com/1424-8220/22/16/6042) on each subject. Calculated the accuracy for each subject and variance in accuracy among subjects to compare with the second approach.
2. Next we applied **EFA & LDA with** [Whitening Technique](https://www.mdpi.com/1424-8220/22/16/6042) on each subject and again calculated accuracy and variance in accuracy among subjects.

## Results
<p align="center">
The accuracies for 9 subjects are : 
</p>
<p align="center">
<img src="https://github.com/KJ-999/Controlling-Machines-with-Human-Brain/blob/main/Assets/Accuracies.png" alt="The accuracies for 9 subjects are : " width="600"/>
</p>

* We can observe that accuracy has been improved and variance has been reduced drastically.
* After whitening the mean of accuracy improved by almost **35%** and variance reduced to **50%** of original variance.

## Other References: 
Yang, Y.M.; Lim, W.; Kim, B.M. Eigenface analysis for brain signal classification: A novel algorithm. Int. J. Telemed. Clin. Pract. 2017, 2, 148–153. [Google Scholar](https://scholar.google.com/scholar_lookup?title=Eigenface+analysis+for+brain+signal+classification:+A+novel+algorithm&author=Yang,+Y.M.&author=Lim,+W.&author=Kim,+B.M.&publication_year=2017&journal=Int.+J.+Telemed.+Clin.+Pract.&volume=2&pages=148%E2%80%93153&doi=10.1504/IJTMCP.2017.083887)


