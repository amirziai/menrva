import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron, LogisticRegression, BayesianRidge, SGDClassifier, \
    PassiveAggressiveClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# more information about sklearn's incremental models
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html
# http://scikit-learn.org/stable/modules/scaling_strategies.html
# TODO add xgboost, keras
# TODO add other models
models_linear = [Perceptron, LogisticRegression, BayesianRidge, LinearSVC]
models_online = [Perceptron, MultinomialNB, SGDClassifier, PassiveAggressiveClassifier]
models_nonlinear_cheap = [DecisionTreeClassifier]
models_nonlinear_expensive = [RandomForestClassifier, SVC, GradientBoostingClassifier]

hyperparameters = {
    LogisticRegression: [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ],
    RandomForestClassifier: {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]},
    
}


def requirements_bare_minimum(y_train):
    # model has to beat:
    # 1- random guessing (AUC must be higher than 0.5), and
    # 2- majority vote classification, important for imbalanced datasets
    # 3- cross-validation variation should not be too high (less than 0.1)

    return {'auc': 0.5,
            'accuracy': pd.Series(y_train).value_counts(normalize=True)[0],
            'accuracy_std': 0.1}


def tune(insights, x_train, x_test, y_train, y_test, models='all', requirements=None, maximize=False):
    if requirements is not None:
        requirements = requirements_bare_minimum(y_train)

    # do vanilla models satisfy the requirements?
    # assuming decision tree is the most intuitive, then logistic regression and then random forest
    # TODO: extend this to metrics other than accuracy using the confusion matrix
    for model_name in ['dt', 'lr', 'rf']:
        model_insights = insights[model_name]
        model_variation = np.std(model_insights['metric_folds'])

        if model_insights['accuracy'] > requirements['accuracy'] and \
                        model_variation < requirements['accuracy_std'] and \
                        model_insights['auc'] > requirements['auc'] and \
                        model_insights['accuracy_test'] > requirements['accuracy']:
            return model_name

    # model selection and tuning loop
    models_to_train = []

    if models == 'all':
        models_to_train += models_linear + models_nonlinear_cheap + models_nonlinear_expensive
    elif models == 'linear':
        models_to_train += models_online
    elif models_to_train == 'cheap':
        models_to_train += models_linear + models_nonlinear_cheap

    for model in models_to_train:
        # TODO: add the looping logic
        
        pass

    return None