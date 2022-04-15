import os
import numpy as np

from src.base.gt_loaders.gen_gt import GenGTLoader
from src.base.gt_loaders.gt_names import GTName
from src.base.gt_loaders.gen_gt import Eval
from src.m_utils import constants as cts


class ColorFeret_GTLoader(GenGTLoader):
    def __init__(self, aligned, ignore_err):
        super().__init__(GTName.COLOR_FERET, aligned, ignore_err, os.path.join(cts.BASE_PATH, 'colorferet', 'dvd2', 'data'))
        
        self.train_aligned_path = os.path.join(self.base_dataset_path, 'aligned_images')
        self.train_not_aligned_path = os.path.join(self.base_dataset_path, 'images')

        
    def setup_data(self):
        path = None
        if self.aligned:
            path = self.train_aligned_path
        else:
            path = self.train_not_aligned_path
        
        self.files_list_train = []
        for root, dirs, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                if self.is_valid_imgpath(p):
                    self.files_list_train.append(p)

        print('len(files_list_train): ', len(self.files_list_train))
        
        # COMPLIANT WITH ICAO REQUISITE
        self.labels_train = np.array([[Eval.COMPLIANT.value] for _ in self.files_list_train])
        
        print('labels_train.shape: {}'.format(self.labels_train.shape))
        
        super().setup_df(labels_names=[cts.ICAO_REQ.VEIL.value])
        super().save_data()
    