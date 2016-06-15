import numpy as np
import pandas as pd


def requirements_bare_minimum(y_train):
    # model has to beat
    # 1- random guessing (AUC must be higher than 0.5), and
    # 2- majority vote classification, important for imbalanced datasets
    # 3- cross-validation variation should not be too high (less than 0.1)

    return {'auc': 0.5,
            'accuracy': pd.Series(y_train).value_counts(normalize=True)[0],
            'accuracy_std': 0.1}


def tune(insights, x_train, x_test, y_train, y_test, requirements=None):
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
