import os
import numpy as np
import pandas as pd

from src.base.gt_loaders.gen_gt import GenGTLoader, Eval
from src.base.gt_loaders.gt_names import GTName
from src.m_utils import constants as cts

class Genki4k_GTLoader(GenGTLoader):
    def __init__(self, aligned, ignore_err):
        super().__init__(GTName.GENKI, aligned, ignore_err, os.path.join(cts.BASE_PATH,'genki4k'))
        
        self.train_aligned_path = os.path.join(f'{self.base_dataset_path}', 'aligned', 'class_name')
        self.train_not_aligned_path = os.path.join(f'{self.base_dataset_path}', 'not_aligned')
        
    
    def setup_data(self):
        if self.aligned:
            self.files_list_train = [os.path.join(self.train_aligned_path, x) for x in sorted(os.listdir(self.train_aligned_path))]
        else:
            self.files_list_train = [os.path.join(self.train_not_aligned_path, x) for x in sorted(os.listdir(self.train_not_aligned_path))]

        print('len(files_list_train): ', len(self.files_list_train))
        
        labels_file = os.path.join(self.base_dataset_path, 'labels.txt')
        df = pd.read_csv(labels_file, sep=' ')
        for f_name in self.files_list_train:
            idx = int(f_name.split('file')[1].split('.')[0])-1
            df.at[idx,'img_name'] = f_name
        
        tmp_df = df[~df.img_name.isna()] # not all images are aligned
        
        self.ground_truth_df = pd.DataFrame(columns=['img_path', 'mouth', 'veil'])
        self.ground_truth_df['img_path'] = tmp_df.img_name.values
        self.ground_truth_df['mouth'] = np.array([[Eval.NON_COMPLIANT.value] if x == 1 else [Eval.COMPLIANT.value] \
                                                  for x in tmp_df.smile.values])
        self.ground_truth_df['veil'] = np.array([[Eval.COMPLIANT.value] for _ in tmp_df.img_name])
        
        super().save_data()
        
        print('Ground Truth shape: ', self.ground_truth_df.shape)
    
    
    