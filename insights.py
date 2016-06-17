import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score


random_state = 0
cv = 5  # number of folds for cross validation


def decision_tree(X, y, regression, max_depth=3):
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    from IPython.core.pylabtools import figsize
    from IPython.display import Image
    figsize(12.5, 6)
    import pydot
    
    if regression:
        clf = DecisionTreeRegressor(max_depth=max_depth)
    else:
        clf = DecisionTreeClassifier(max_depth=max_depth)
        
    clf.fit(X, y)
    dot_data = StringIO()  
    export_graphviz(clf, out_file=dot_data, feature_names=list(X.columns),
                    filled=True, rounded=True,)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())


def viz(clf, clf_raw):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    
    print 'Accuracy: %s' % clf.score(x_test, y_test)
    
    try:
        featureImportance(x_train, clf, 0.01)
    except:
        pass
        
    plot_learning_curve(clf_raw,
                        'Learning curves', x_train, y_train, cv=5)
    y_pred = clf.predict(x_test)
    print classification_report(y_test, y_pred)

    plot_roc_curve(clf, (x_train, y_train, x_test, y_test), False)
    precision_recall_curve(clf, x_test, y_test)

    statistics(clf, x_train, y_train, x_test, y_test, False)

    plot_confusion_matrix(clf, x_test, y_test)
    print confusion_matrix(y_test, y_pred)

    # instances
    instances= {'good': [], 'bad': []}
    preds_proba = clf.predict_proba(x_test)
    preds = clf.predict(x_test)

    l = []
    for i in range(len(y_test)):
        correct_index = list(y_test)[i]
        pred = preds[i]
        pred_proba = preds_proba[i][pred]

        instances[('good' if pred == correct_index else 'bad')].append(pred_proba)

        l.append({'Type': ('Correct' if pred == correct_index else 'Incorrect'),
                 'Outcome': le.inverse_transform(correct_index),
                 'Probability': pred_proba})

    df_sns_proba = pd.DataFrame(l)

    pd.Series(instances['good']).hist()
    pd.Series(instances['bad']).hist()
    plt.show()
    l = [{'Type': 'Correct', 'Probability': x} for x in instances['good']]
    l += [{'Type': 'Incorrect', 'Probability': x} for x in instances['bad']]
    df_sns = pd.DataFrame(l)
    sns.violinplot(x='Type', y='Probability', data=df_sns)
    plt.show()
    sns.boxplot(x='Type', y='Probability', data=df_sns)
    plt.show()
    sns.boxplot(x='Outcome', y='Probability', hue='Type', data=df_sns_proba)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    probabilities_to_check = np.linspace(0, 1, num=11)

    l = []
    for probability_threshold in probabilities_to_check:
        a = [list(y_test)[i] == preds[i] for i, p in enumerate(preds_proba) if np.max(p) >= probability_threshold]
        accuracy = np.nan
        if len(a) > 0:
            accuracy = sum(a) / float(len(a))
            
        l.append({'Probability threshold': probability_threshold,
                 'Percentage of cases': len(a) / float(len(y_test)),
                  'Accuracy': accuracy
                 })

    a = pd.DataFrame(l)[['Probability threshold', 'Percentage of cases', 'Accuracy']]
    print a

    plt.plot(a['Probability threshold'], a['Percentage of cases'])
    plt.plot(a['Probability threshold'], a['Accuracy'], c='g')
    plt.ylim([0, 1])
    plt.show()
    plt.plot(a['Probability threshold'], a['Percentage of cases'].diff().cumsum())
    plt.plot(a['Probability threshold'], a['Accuracy'].diff().cumsum())
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    from sklearn.learning_curve import learning_curve
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def plot_roc_curve(clf, data, regression):
    from sklearn.metrics import roc_curve, auc
    
    x_train, y_train, x_test, y_test = data
    
    if regression:
        """
        plots actual vs. predicted
        plots error
        """
        pred_train = clf.predict(x_train)
        pred_test = clf.predict(x_test) 
        plt.scatter(y_test, pred_test, color='r', label='X')
        plt.scatter(y_train, pred_train)
        plt.plot([min(min(pred_train), min(pred_test)), max(max(pred_train), max(pred_test))],
                 [min(min(y_train), min(y_test)), max(max(y_train), max(y_test))], 'k--')
        plt.title('Predicted vs. actual for train and test')
        plt.legend()
        plt.show()
        
        print '\n'
        
        plt.plot([x - y for x, y in zip(y_test, pred_test)], color='r')
        plt.plot([x - y for x, y in zip(y_train, pred_train)])
        plt.title('Actual - prediction (error)')
        plt.show()
    else:
        """
        plots the ROC curve
        """
        pred_train_prob = [x[1] for x in clf.predict_proba(x_train)]
        pred_test_prob = [x[1] for x in clf.predict_proba(x_test)]
        fpr, tpr, thresholds = roc_curve(y_train, pred_train_prob)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b',
        label='AUC = %0.2f'% roc_auc)
        fpr, tpr, thresholds = roc_curve(y_test, pred_test_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'r',
                label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
        
def plot_confusion_matrix(clf, x_test, y_test, title='Confusion matrix', cmap=plt.cm.Blues):
    from sklearn.metrics import confusion_matrix
    
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
def precision_recall_curve(clf, x_test, y_test):
    from sklearn.metrics import precision_recall_curve
    
    for i in range(2):
        y_probabilities = [x[i] for x in clf.predict_proba(x_test)]
        precision, recall, thresholds = precision_recall_curve(y_test, y_probabilities)

        plt.title('Precision Recall Curve')
        plt.plot(recall, precision, 'b')

    plt.show()


def barh_dic(f, title=None):
    import operator
    
    y = sorted(f.items(), key=operator.itemgetter(1))
    keys = [a[0] for a in y]
    vals = [a[1] for a in y]
    plt.barh(range(len(y)), vals, align='center')
    plt.yticks(range(len(y)), keys)
    
    if title:
        plt.title(title)
        
    plt.show()


def feature_importance(X, clf, threshold=0.03, return_=False, show=True):
    item = clf.feature_importances_
        
    val = dict((x, y) for x, y in zip(X.columns, item))
    
    val_ = dict({k:val[k] for k in val if val[k] >= threshold})
        
    if show:
        barh_dic(val_, 'Feature importance')

    if return_:
        return val
    
    
def statistics(clf, x_train, y_train, x_test, y_test, regression):
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import roc_auc_score
    
    # if regression:
    #     r2_train = clf.score(x_train, y_train)
    #     r2_test = clf.score(x_test, y_test)
    #     mse_train = mse(y_train, clf.predict(x_train))
    #     mse_test = mse(y_test, clf.predict(x_test))
    #     mae_train = mae(y_train, clf.predict(x_train))
    #     mae_test = mae(y_test, clf.predict(x_test))
    #     # mean relative error
    #     mre_train = mean_relative_error(y_train, clf.predict(x_train))
    #     mre_test = mean_relative_error(y_test, clf.predict(x_test))
    #
    #     return pd.DataFrame({
    #             'R2 train': [r2_train],
    #             'R2 test': [r2_test],
    #             'R2 %': [r2_test / r2_train - 1],
    #             'MSE train': [mse_train],
    #             'MSE test': [mse_test],
    #             'MSE %': mse_test / mse_train - 1,
    #             'MAE train': [mae_train],
    #             'MAE test': [mae_test],
    #             'MAE %': mae_test / mae_train - 1,
    #             'MRE train': [mre_train],
    #             'MRE test': [mre_test],
    #             'MRE %': mre_test / mre_train - 1
    #         }).transpose()
    #
    # else:
    accuracy_train = clf.score(x_train, y_train)
    accuracy_test = clf.score(x_test, y_test)
    precision_train = precision_score(y_train, clf.predict(x_train))
    precision_test = precision_score(y_test, clf.predict(x_test))
    recall_train = recall_score(y_train, clf.predict(x_train))
    recall_test = recall_score(y_test, clf.predict(x_test))
    f1_train = f1_score(y_train, clf.predict(x_train))
    f1_test = f1_score(y_test, clf.predict(x_test))

    roc_train = -1
    roc_test = -1
    if hasattr(clf, 'predict_proba'):
        roc_train = roc_auc_score(y_train, [x[1] for x in clf.predict_proba(x_train)])
        roc_test = roc_auc_score(y_test, [x[1] for x in clf.predict_proba(x_test)])

    val = {
            'Accuracy train': [accuracy_train],
            'Accuracy test': [accuracy_test],
            'Accuracy %': accuracy_test / accuracy_train - 1,
            'Precision train': [precision_train],
            'Precision test': [precision_test],
            'Precision %': precision_test / precision_train - 1,
            'Recall train': [recall_train],
            'Recall test': [recall_test],
            'Recall %': recall_test / recall_train - 1,
            'F1 train': [f1_train],
            'F1 test': [f1_test],
            'F1 %': f1_test / f1_train - 1,
    }

    if roc_train != -1:
        val['ROC train'] = [roc_train]
        val['ROC test'] = [roc_test]
        val['ROC %'] = roc_test / roc_train - 1

    return pd.DataFrame(val).transpose()


def clf_predict_proba(clf, x):
    return [_[1] for _ in clf.predict_proba(x)]


def clf_scores(clf, x_train, y_train, x_test, y_test):
    info = dict()

    # TODO: extend this to a confusion matrix per fold for more flexibility downstream (tuning)
    # TODO: calculate a set of ROC curves per fold instead of running it on test, currently introducing bias
    scores = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_test_predicted = clf.predict(x_test)
    info['accuracy'] = min(scores)
    info['accuracy_test'] = accuracy_score(y_test, y_test_predicted)
    info['accuracy_folds'] = scores
    info['confusion_matrix'] = confusion_matrix(y_test, y_test_predicted)
    clf.fit(x_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, clf_predict_proba(clf, x_test))
    info['fpr'] = fpr
    info['tpr'] = tpr
    info['auc'] = auc(fpr, tpr)

    return info


def generate_insights(scores):
    print 'AUC curves'

    for model in scores:
        fpr = scores[model]['fpr']
        tpr = scores[model]['tpr']
        auc_score = scores[model]['auc']
        plt.plot(fpr, tpr, label='AUC %s = %0.2f' % (model, auc_score))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print 'Metric over fold distribution'

    metric_distribution = {k: scores[k]['accuracy_folds'] for k in scores.keys()}
    pd.DataFrame(metric_distribution).boxplot(return_type='dict');


def fit_vanilla(x_train, x_test, y_train, y_test):
    scores = dict()

    # Decision tree
    dt = DecisionTreeClassifier(random_state=random_state)
    scores['dt'] = clf_scores(dt, x_train, y_train, x_test, y_test)

    # Logistic Regression
    lr = LogisticRegression(random_state=random_state, n_jobs=-1)
    scores['lr'] = clf_scores(lr, x_train, y_train, x_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    scores['rf'] = clf_scores(rf, x_train, y_train, x_test, y_test)

    return scores
