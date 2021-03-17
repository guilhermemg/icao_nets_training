import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from gt_loaders.gen_gt import Eval


def __draw_roc_curve(fpr, tpr, eer, th):
    fig = plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve - ' + 'EER: {:.4f} | Thresh: {:.4f}'.format(eer, th))
    plt.show()
    return fig


def __draw_far_frr_curve(th_range, frr, far, eer):
    fig = plt.figure(1)
    plt.plot(th_range, frr,'-r')
    plt.plot(th_range, far,'-b')
    plt.xlabel('Threshold')
    plt.ylabel('FAR/FRR %')
    plt.xlim([0, 1.02])
    plt.ylim([0, 100])
    plt.title(f'EER = {round(eer,4)}')
    plt.legend(['FRR','FAR'], loc='upper center')
    plt.show()
    return fig


def __calculate_far(y_true, y_pred):
    far = []
    n_attacks = len([x for x in y_true if x == Eval.NON_COMPLIANT.value])
    th_range = np.arange(0, 1.02, 0.01) 
    for th in th_range:
        num = 0
        for tr_val,pred in zip(y_true,y_pred):
            if pred >= th and tr_val == Eval.NON_COMPLIANT.value:
                num += 1
        far.append(round((num/n_attacks) * 100, 2))

    far = np.array(far) 
    return far

    
def __calculate_frr(y_true, y_pred):
    frr = []
    n_reals = len([x for x in y_true if x == Eval.COMPLIANT.value])
    th_range = np.arange(0, 1.02, 0.01) 
    for th in th_range:
        num = 0
        for tr_val,pred in zip(y_true,y_pred):
            if pred < th and tr_val == Eval.COMPLIANT.value:
                num += 1
        frr.append(round((num/n_reals) * 100, 2))

    frr = np.array(frr)    
    return frr


def calculate_eer(y_true, y_pred, req):
    fpr, tpr, ths = roc_curve(y_true, y_pred)
    far = __calculate_far(y_true, y_pred)
    frr = __calculate_frr(y_true, y_pred)
    th_range = np.arange(0, 1.02, 0.01)
    
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    best_th = interp1d(fpr, ths)(eer)

    roc_curve_fig = __draw_roc_curve(fpr, tpr, eer, best_th)
    far_frr_curve_fig = __draw_far_frr_curve(th_range, far, frr, eer)
    
    eer = round(eer*100, 4)
    print(f'Requisite: {req} - EER: {eer}% - Best Threshold: {best_th}')

    return eer, best_th, roc_curve_fig, far_frr_curve_fig


def get_classification_report(test_gen, y_true, y_pred):
    print('Classification report -----------------------------------')
        
    print(classification_report(y_true=y_true, 
                                y_pred=y_pred, 
                                target_names=list(test_gen.class_indices.keys()), 
                                labels=list(test_gen.class_indices.values())))

def calculate_accuracy(y_true, y_pred):
    print('Accuracy ------------------------------------------------')
    acc = round(accuracy_score(y_true, y_pred)*100, 2)
    print(f'Model Accuracy: {acc}%')
    print('---------------------------------------------------------')
    return acc
    

def get_confusion_matrix(y_true, y_pred):
    print('Confusion matrix ----------------------------------------')
    TN,FP,FN,TP = confusion_matrix(y_true, y_pred, labels=[Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]).ravel()
    print(f'TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}')
    return TN,FP,FN,TP