# menrva
A Python platform for supervised machine learning

### A three part platform
1. Merge, reshape, extract feature
2. Pre-processing, training and evaluation
3. Model management, developing this piece

### 1- Data wrangling
Easy merging, reshaping and automatic feature engineering.

### 2- Modeling
Supports regression and classification. Does things like one-hot encoding, label encoding, and null-value imputation. Then progressively trains models from simple to complex and performs grid search for hyperparameter optimization. Finally reports model quality statistics, supporting reports and visualization (ROC curve, contingency matrix, etc.) and exports a pickled model.

### 3- Model management
Persists new models to disk and then uses Redis to serve models from memory. Logs model usage for experimentation and debugging purposes.

### TODO
- add a dropzone UI to drag and drop pickled models
- get a list of all models (in memory and on disk)
- Log usage of models

### MIT License