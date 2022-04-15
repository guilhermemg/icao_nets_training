import os
import numpy as np

from src.base.gt_loaders.gen_gt import GenGTLoader
from src.base.gt_loaders.gt_names import GTName
from src.m_utils import constants as cts

class PybFVCGTLoader(GenGTLoader):
    def __init__(self, name, ground_truth_path, aligned, ignore_err):
        self.train_aligned_path = f'{cts.ICAO_DATASET_PATH}/train_align/class_name'
        self.valid_aligned_path = f'{cts.ICAO_DATASET_PATH}/validation_align/class_name'
        self.train_not_aligned_path = f'{cts.ICAO_DATASET_PATH}/train'
        self.valid_not_aligned_path = f'{cts.ICAO_DATASET_PATH}/validation'
        
        self.ground_truth_path = ground_truth_path
        
        super().__init__(name, aligned, ignore_err, cts.ICAO_DATASET_PATH)

    
    def setup_data(self):
        if self.aligned:
            self.files_list_train = [os.path.join(self.train_aligned_path, x) for x in sorted(os.listdir(self.train_aligned_path))]
            self.files_list_val = [os.path.join(self.valid_aligned_path, x) for x in sorted(os.listdir(self.valid_aligned_path))]
        else:
            self.files_list_train = [os.path.join(self.train_not_aligned_path, x) for x in sorted(os.listdir(self.train_not_aligned_path))]
            self.files_list_val = [os.path.join(self.valid_not_aligned_path, x) for x in sorted(os.listdir(self.valid_not_aligned_path))]

        print('len(files_list_train): ', len(self.files_list_train))
        print('len(files_list_val): ', len(self.files_list_val))

        for file_id in self.files_list_train:
            self.labels_train.append(self.__get_labels(file_id, 'train'))
        for file_id in self.files_list_val:
            self.labels_val.append(self.__get_labels(file_id, 'validation')) 
        
        print('labels_train.shape: ({},{})'.format(len(self.labels_train), len(self.labels_train[0])))
        print('labels_val.shape: ({},{})'.format(len(self.labels_val), len(self.labels_val[10])))
        
        super().setup_df()
        super().save_data()
    
    
    def __get_labels(self, file_id, orig=''):
        labels_file = ''
        if orig != '' and self.name == GTName.FVC:
            labels_file = self.ground_truth_path + orig + '/' + self.__clean_fid(file_id) + '.mrk'
        else:
            labels_file = self.ground_truth_path + self.__clean_fid(file_id) + '.mrk'
        try:
            with open(labels_file) as labels_file:
                lines = labels_file.readlines()
                lines = [int(x.replace('\n','')) for x in lines[2:]]
                return np.array(lines)
        except:
            if not self.ignore_err:
                print('File not found: ' + labels_file)
            
    
    def __clean_fid(self, fid):
        fid = fid.split('/')[-1]
        fid = fid.replace('.jpg','').replace('.png','').replace('.JPG','').replace('.PNG','')
        return fid