import os
import numpy as np
import pandas as pd

from src.base.gt_loaders.gen_gt import GenGTLoader
from src.base.gt_loaders.gt_names import GTName
from src.base.gt_loaders.gen_gt import Eval
from src.m_utils import constants as cts


class IMFD_GTLoader(GenGTLoader):
    def __init__(self, aligned, ignore_err):
        super().__init__(GTName.IMFD, aligned, ignore_err, os.path.join(cts.BASE_PATH,'imfd'))
        
        self.train_aligned_path = os.path.join(self.base_dataset_path, 'aligned', 'class_name')
        self.train_not_aligned_path = os.path.join(self.base_dataset_path, 'not_aligned')

        
    def setup_data(self):
        if self.aligned:
            self.files_list_train = [os.path.join(self.train_aligned_path, x) for x in sorted(os.listdir(self.train_aligned_path))]
        else:
            self.files_list_train = [os.path.join(self.train_not_aligned_path, x) for x in sorted(os.listdir(self.train_not_aligned_path))]

        print('len(files_list_train): ', len(self.files_list_train))
        
        self.ground_truth_df = pd.DataFrame(columns=['img_path', 'mouth', 'veil'])
        self.ground_truth_df['img_path'] = self.files_list_train
        self.ground_truth_df['mouth'] = np.array([[Eval.DUMMY.value] for _ in self.files_list_train])
        self.ground_truth_df['veil'] = np.array([[Eval.NON_COMPLIANT.value] for _ in self.files_list_train])
        
        super().save_data()
        
        print('Ground Truth shape: ', self.ground_truth_df.shape)
    