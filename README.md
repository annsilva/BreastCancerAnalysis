# BreastCancerAnalysis

## Overview
This project focuses on analyzing the Breast Cancer Wisconsin (Diagnostic) dataset using machine learning techniques. The goal is to classify breast cancer tumors as malignant or benign based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which includes the following:
569 instances
30 numeric, predictive attributes
2 classes: Malignant and Benign

## Features
The features in the dataset describe characteristics of cell nuclei present in the images, including:
Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave points
Symmetry
Fractal dimension

For each feature, the mean, standard error, and "worst" (mean of the three largest values) are computed, resulting in 30 features.

## Project Structure
The project includes the following main components:
Data loading and preprocessing
Exploratory data analysis
Feature selection
Model training and evaluation
Hyperparameter Tuning

## Files
scaler.pkl: Pickle file containing the StandardScaler object
selector.pkl: Pickle file containing the SelectKBest feature selector

## Results
The project includes visualizations of feature importance and model performance metrics.

## References

UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Data Set
