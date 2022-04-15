import os
import abc
import pandas as pd
import numpy as np
import seaborn as sns

from enum import Enum

import src.m_utils.constants as cts


class Eval(Enum):
    COMPLIANT = 1
    NON_COMPLIANT = 0
    DUMMY = -1
    DUMMY_CLS = 2   # dummy value used in classification (-1 is no allowed)
    NO_ANSWER = -99

    

class GenGTLoader(metaclass=abc.ABCMeta):
    def __init__(self, name, aligned, ignore_err, base_dataset_path):
        self.name = name
        self.aligned = aligned
        self.ignore_err = ignore_err
        self.base_dataset_path = base_dataset_path
        
        self.files_list_train = []
        self.files_list_val = []
        self.labels_train = []
        self.labels_val = []
        
        self.ground_truth_df = None
    
    
    def get_name(self):
        return self.name
    
    def is_aligned(self):
        return self.aligned
    
    
    def setup_df(self, labels_names=None):
        print(f'-- GT: {self.get_name()} --')
        
        if labels_names is None:
            labels_names = self.get_icao_labels().label_name.values.tolist()
        
        cols = ['img_path'] + labels_names
        self.ground_truth_df = pd.DataFrame(columns=cols)
        self.ground_truth_df['img_path'] = self.files_list_train + self.files_list_val
        
        labels_list = []
        if self.labels_val != []:
            labels_list = self.labels_train + self.labels_val
        else:
            labels_list = self.labels_train
            
        labels_list = [np.array([Eval.NO_ANSWER.value for i in range(23)]) if l is None else l for l in labels_list]
        
        labels_mat = np.stack(labels_list, axis=0)
        
        print('Ground Truth shape: ', labels_mat.shape)
        
        for idx,req in enumerate(labels_names):
            if idx < labels_mat.shape[1]:
                self.ground_truth_df[req] = labels_mat[:,idx]
        
    
    def count_images_per_class(self):
        labels_names = self.ground_truth_df.columns.tolist()
        labels_names.remove('img_path')
        cols = ['Compliant','Dummy','NonCompliant','NoAnswer','Total',
                'CompPerc','DummyPerc','NonCompPerc','NoAnsPerc','TotalPerc']
        final_df = pd.DataFrame(columns=cols, index=labels_names)
        total = self.ground_truth_df.shape[0]
        for label in labels_names:
            comp = self.ground_truth_df[self.ground_truth_df[label] == Eval.COMPLIANT.value][label].count()
            dummy = self.ground_truth_df[self.ground_truth_df[label] == Eval.DUMMY.value][label].count()
            non_comp = self.ground_truth_df[self.ground_truth_df[label] == Eval.NON_COMPLIANT.value][label].count()
            no_answer = self.ground_truth_df[self.ground_truth_df[label] == Eval.NO_ANSWER.value][label].count()
            
            comp_perc = round(comp/total * 100, 2)
            non_comp_perc = round(non_comp/total * 100, 2)
            dummy_perc = round(dummy/total * 100, 2)
            no_answer_perc = round(no_answer/total * 100, 2)
            
            final_df.at[label, 'Compliant'] = comp
            final_df.at[label, 'Dummy'] = dummy
            final_df.at[label, 'NonCompliant'] = non_comp
            final_df.at[label, 'NoAnswer'] = no_answer
            final_df.at[label, 'Total'] = comp + dummy + non_comp + no_answer
            final_df.at[label, 'CompPerc'] = comp_perc
            final_df.at[label, 'NonCompPerc'] = non_comp_perc
            final_df.at[label, 'DummyPerc'] = dummy_perc
            final_df.at[label, 'NoAnsPerc'] = no_answer_perc
            final_df.at[label, 'TotalPerc'] = comp_perc + non_comp_perc + no_answer_perc + dummy_perc
            
        final_df = final_df.astype({'CompPerc':np.float64, 'NonCompPerc':np.float64, 
                                    'DummyPerc':np.float64, 'NoAnsPerc':np.float64, 'TotalPerc':np.float64})    
        
        cm = sns.light_palette("green", as_cmap=True)
        return final_df.style.background_gradient(cmap=cm)

    
    def get_icao_labels(self):
        return pd.read_csv(cts.ICAO_CLASSES_PATH)
    
    def is_valid_imgpath(self, imgpath):
        return '.JPG' in imgpath or '.PNG' in imgpath \
                or '.jpg' in imgpath or '.png' in imgpath \
                or '.ppm' in imgpath
    
    
    def load_data(self):
        dtypes_dict = {x: int for x in cts.ICAO_REQ.list_reqs_names()}
        if self.aligned:
            #self.ground_truth_df = pd.read_csv(os.path.join(self.base_dataset_path, f'ground_truth_aligned.csv'))
            self.train_df = pd.read_csv(os.path.join(self.base_dataset_path, f'{self.name.value}_aligned_train.csv'), dtype=dtypes_dict)
            self.validation_df = pd.read_csv(os.path.join(self.base_dataset_path, f'{self.name.value}_aligned_validation.csv'), dtype=dtypes_dict)
            self.test_df = pd.read_csv(os.path.join(self.base_dataset_path, f'{self.name.value}_aligned_test.csv'), dtype=dtypes_dict)
        else:
            #self.ground_truth_df = pd.read_csv(os.path.join(self.base_dataset_path, f'ground_truth_not_aligned.csv'))
            self.train_df = pd.read_csv(os.path.join(self.base_dataset_path, f'{self.name.value}_not_aligned_train.csv'), dtype=dtypes_dict)
            self.validation_df = pd.read_csv(os.path.join(self.base_dataset_path, f'{self.name.value}_not_aligned_validation.csv'), dtype=dtypes_dict)
            self.test_df = pd.read_csv(os.path.join(self.base_dataset_path, f'{self.name.value}_not_aligned_test.csv'), dtype=dtypes_dict)
    
    
    def save_data(self):
        if self.aligned:
            self.ground_truth_df.to_csv(os.path.join(self.base_dataset_path, 'ground_truth_aligned.csv'), index=False)
        else:
            self.ground_truth_df.to_csv(os.path.join(self.base_dataset_path, 'ground_truth_not_aligned.csv'), index=False)
        