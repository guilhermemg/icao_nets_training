import cv2
import numpy as np
import pandas as pd

from enum import Enum

import matplotlib.pyplot as plt

from neptune.new.types import File

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from gt_loaders.gen_gt import Eval

import utils.constants as cts
import utils.draw_utils as dr

from utils.constants import SEED


class DataSource(Enum):
    VALIDATION = 'validation'
    TEST = 'test'  

    
class DataPredSelection(Enum):
    ANY = {'title': 'Any (TP,FP,TN,FN) images', 'abv': 'any_imgs'}
    ONLY_FP = {'title': 'Only False Positives images', 'abv': 'fp_only'}
    ONLY_FN = {'title': 'Only False Negatives images', 'abv': 'fn_only'}
    ONLY_TP = {'title': 'Only True Positives images', 'abv': 'tp_only'}
    ONLY_TN = {'title': 'Only True Negatives images', 'abv': 'tn_only'}
    

class VsoftEvaluator:
    def __init__(self, data_src, img_paths, y_true, y_hat, requisite):
        print('Creating VsoftEvaluator')
        print(f' .. Data source: {data_src.value.upper()}')
        print(f' .. Requisite: {requisite.value.upper()}')
        print(f' .. n img_paths: {len(img_paths)}')
        print(f' .. n y_true: {len(y_true)}')
        print(f' .. n y_hat: {len(y_hat)}')
        
        self.img_paths = img_paths
        self.y_true = y_true
        self.y_hat = y_hat
        self.y_hat_discrete = np.where(self.y_hat < 0.5, 0, 1)
        self.data_src = data_src
        self.requisite = requisite
        

    def __get_classification_report(self):
        print('Classification report -----------------------------------')
        
        #target_names = list(data_gen.class_indices.keys())
        #labels = list(data_gen.class_indices.values())
        
        print(classification_report(y_true=self.y_true, 
                                    y_pred=self.y_hat_discrete))
                                    #target_names=target_names, 
                                    #labels=labels))

    def __calculate_accuracy(self):
        print('Accuracy ------------------------------------------------')
        ACC = round(accuracy_score(self.y_true, self.y_hat_discrete), 4)
        print(f'Model Accuracy: {ACC*100}%')
        print('---------------------------------------------------------')


    def __get_confusion_matrix(self):
        print('Confusion matrix ----------------------------------------')
        TN,FP,FN,TP = confusion_matrix(self.y_true, self.y_hat_discrete, 
                                       labels=[Eval.NON_COMPLIANT.value, Eval.COMPLIANT.value]).ravel()
        FAR = round(FP/(FP+TN),4)
        FRR = round(FN/(FN+TP),4)
        EER_mean = round((FAR+FRR)/2.,4)
        
        print(f'FAR: {FAR*100}% | FRR: {FRR*100}% | EER_mean: {EER_mean*100}% | TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}')
        
    
    def calculate_metrics(self):
        print('================================================')
        print('Testing VSOFT BiopassID ICAO CHECK')
        print(f'Requisite: {self.requisite.value.upper()}')
        
        self.__get_classification_report()
        self.__get_confusion_matrix()
        self.__calculate_accuracy()
        
        print('================================================')
    
    

    def __select_viz_data(self, n_imgs, data_pred_selection):
        tmp_df = pd.DataFrame()
        tmp_df['img_name'] = self.img_paths
        tmp_df['comp'] = self.y_true.astype(int)
        tmp_df['pred'] = self.y_hat_discrete.astype(int)
        
        data_src_uppercase = self.data_src.value.upper()
        data_src_lowercase = self.data_src.value.lower()
        
        COMP = Eval.COMPLIANT.value#int(Eval.COMPLIANT.value)
        NON_COMP = Eval.NON_COMPLIANT.value#int(Eval.NON_COMPLIANT.value)
        
        viz_title = None
        if data_pred_selection.name == DataPredSelection.ONLY_FN.name:
            tmp_df = tmp_df[(tmp_df.comp == COMP) & (tmp_df.pred == NON_COMP)]
        elif data_pred_selection.name == DataPredSelection.ONLY_FP.name:
            tmp_df = tmp_df[(tmp_df.comp == NON_COMP) & (tmp_df.pred == COMP)]
        elif data_pred_selection.name == DataPredSelection.ONLY_TP.name:
            tmp_df = tmp_df[(tmp_df.comp == COMP) & (tmp_df.pred== COMP)]
        elif data_pred_selection.name == DataPredSelection.ONLY_TN.name:
            tmp_df = tmp_df[(tmp_df.comp == NON_COMP) & (tmp_df.pred == NON_COMP)]
        
        n_imgs = tmp_df.shape[0] if tmp_df.shape[0] < n_imgs else n_imgs
        tmp_df = tmp_df.sample(n=n_imgs, random_state=SEED)
        
        viz_title = f"{data_pred_selection.value['title']} - {self.data_src.value.upper()}" 
        
        return tmp_df, viz_title
    
    
    def __get_img_name(self, img_path):
            return img_path.split("/")[-1].split(".")[0]
    
    
    # sort n samples from data_df based on criteria DataPredSelection
    def vizualize_predictions(self, n_imgs, data_pred_selection=DataPredSelection.ANY):
        tmp_df,viz_title = self.__select_viz_data(n_imgs, data_pred_selection)
        
        labels = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        
        preds = [f'GT: COMP\n {self.__get_img_name(path)}' if x == Eval.COMPLIANT.value else f'GT: NON_COMP\n {self.__get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        
        imgs = [cv2.resize(cv2.imread(img), (224,224)) for img in tmp_df.img_name.values]
        
        f = dr.draw_imgs(imgs, title=viz_title, labels=labels, predictions=preds)
        
    