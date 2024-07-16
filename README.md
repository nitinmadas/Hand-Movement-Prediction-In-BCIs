# Hand-Movement-Prediction-In-BCIs

This repository provides the implemention code for the Brain Computer Interface Channel Whitening [(BCICW)](https://www.mdpi.com/1424-8220/22/16/6042) proposed in the paper:

Choi, H.; Park, J.; Yang, Y.-M. Whitening Technique Based on Gram–Schmidt Orthogonalization for Motor Imagery Classification of Brain-Computer Interface Applications. Sensors 2022, 22, 6042. https://doi.org/10.3390/s22166042

#

In BCI research, major challenges are
- Accurate classification of different mental activities (Motor Imagery)
- Accuracy Consistency across subjects by addressing the variance in brain signals

Our Solution:
- Apply Eigen Face Analysis (EFA) and Linear Discriminant Analysis (LDA) for mental activity( Motor Imagery) classification
- Apply Brain-Computer Interface Channel Whitening (BCICW) for signal processing before feature engineering and classification

Dataset:
Epoched Electroencephalogram (EEG) signal data for channels C3, CZ, and C4 of 9 subjects ([link](https://www.kaggle.com/competitions/ucsd-neural-data-challenge))


# Workflow chart


<p align="center">
The components of the BCICW model
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/KJ-999/Controlling-Machines-with-Human-Brain/main/Assets/Workflow_BCI.webp?token=GHSAT0AAAAAACOLUMH7RMFHSR7Q5HDTYFDWZOSJHQA" alt="The components of the proposed BCICW model" width="400"/>
</p>
