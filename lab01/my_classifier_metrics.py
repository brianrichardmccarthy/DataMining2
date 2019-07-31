import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import interp
from sklearn.model_selection  import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

accuracy = lambda cm: (cm[1,1]+cm[0,0]) / cm.sum()
precision = lambda cm: cm[1,1] / (cm[1,1]+cm[0,1])
recall = lambda cm: cm[1,1] / (cm[1,1]+cm[1,0])
f1_score = lambda cm: 2 * (precision(cm)*recall(cm)) / (precision(cm)+recall(cm))

def cvClassifier(X, y, classifier, n_splits=5, shuffle=False, random_state=None, 
                 show_all_curves=True, 
                 show_chance=True, show_std=True, show_full_legend=False):
    """Wrapper funciton to simplift generation of ROC plot when using cross-validation (with StratifiedKFold).
    """

    # TODO - allow other fold options here (StratifiedShuffleSplit)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    tprs = []   # true positive stats
    aucs = []   # auc states
    cm = np.zeros((2,2))    # confusion matrix
        
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    
    i = 0
    for train, test in cv.split(X, y):
    
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict(X[test])
        y_prob = classifier.predict_proba(X[test])

        cm += confusion_matrix(y[test], y_pred)
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], y_prob[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        label = 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc) if show_full_legend else None
        if show_all_curves: ax.plot(fpr, tpr, lw=1, alpha=0.3, label=label)

        i += 1
        
    if show_chance:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    if show_std:
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic - cv kfold(%d)' % n_splits)
    ax.legend(loc="lower right")
    
    cm = cm/n_splits

    metrics = {"accuracy":accuracy(cm), "precision":precision(cm), "recall":recall(cm), "f1_score":f1_score(cm)}
    print('Accuracy: {:.2f}%'.format(100.0 *  metrics["accuracy"] ))
    print('Precision: {:.2f}%'.format(100.0 * metrics["precision"] ))
    print('Recall:   {:.2f}%'.format(100.0 * metrics["recall"] ))
    print('F1 score:   {:.2f}%'.format(100.0 * metrics["f1_score"] ))

    return ax, cm, metrics


def confusionMatrixPlot(cm,name):
    cm_df = pd.DataFrame(cm, ("No "+name, name), ("No "+name, name))
    ax = sns.heatmap(cm_df, annot=True, annot_kws={"size": 20}, fmt=".1f")
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax

if __name__ == "__main__":
    print("TODO: test code for these functions")