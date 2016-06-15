# menrva
A Python platform for supervised machine learning

### A three layer platform
1. Wrangling: merge, reshape, extract features ** not currently available
2. Modeling: pre-processing, training and evaluation) ** under development
3. Serving: layer- serve and manage models ** serving is available

### 1- Data wrangling
Easy merging, reshaping and automatic feature engineering.

### 2- Modeling
Supports regression and classification. Does one-hot encoding, label encoding, and null-value imputation. Then progressively trains models from simple to complex and performs hyperparameter tuning and model selection in parallel. Finally reports model quality statistics, diagnostic reports (ROC curve, contingency matrix, etc.) and serializes the best model.

### 3- Model management
Persists new models to disk and then uses Redis to serve models from memory. Logs model usage for experimentation and debugging purposes.

### Dependencies
1. Numpy
2. Pandas
3. Scikit-learn

### Installation (ubuntu)
1. Anaconda
1. Redis https://www.digitalocean.com/community/tutorials/how-to-install-and-use-redis
2. Sqlite

Alternatively you can use this image: coming soon.

### TODO
- Finish the flow for the modeling layer
- Create an AWS image for the serving layer
- Log model statistics and create a UI for managing models

### MIT License