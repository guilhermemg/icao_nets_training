import os
import numpy as np
import pandas as pd

from enum import Enum


class DLName(Enum):
    VGGFACE2 = 'vgg' 
    CALTECH = 'caltech'
    FVC_PYBOSSA = 'fvc_pybossa'
    CVL = 'cvl'
    FEI_DB = 'fei'
    COLOR_FERET = 'color_feret'
    GEORGIA_TECH = 'georgia_tech'
    UNI_ESSEX = 'uni_essex'
    ICPR04 = 'icpr04'
    IMFDB = 'imfdb'
    IJBC = 'ijbc'
    LFW = 'lfw'
    CELEBA = 'celeba'
    CASIA_WF = 'casia_webface'
    GENKI4K_DB = 'genki4k'
    
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class DataLoader:
    def __init__(self, name, aligned, dataset_path, restricted_env, max_imgs=None, is_ground_truth=False):
        self.name = name
        self.aligned = aligned
        self.dataset_path = dataset_path
        self.restricted_env = restricted_env
        self.max_imgs = max_imgs
        self.is_ground_truth = is_ground_truth
        
        self.output_path = None
        self.train_dirs_paths = None
        self.test_dirs_paths = None
        self.data_df = None
    
    def get_name(self):
        return self.name
    
    def get_data_df(self):
        return self.data_df
    
    def is_aligned(self):
        return self.aligned
    
    def is_restricted_env(self):
        return self.restricted_env
    
    def is_ground_truth(self):
        return self.is_ground_truth
    
    def __str__(self):
        if self.aligned:
            return self.get_name().value + '_aligned'
        else:
            return self.get_name().value + '_not_aligned'
    
    def set_dirs_paths(self):
        pass

    def set_output_path(self):
        aligned = 'aligned' if self.is_aligned() else 'not_aligned'
        self.output_path = f'{self.dataset_path}/{aligned}_images_names.csv'
    
    def load_data_df(self):
        if os.path.exists(self.output_path):
            self.data_df = pd.read_csv(self.output_path)
    
    def __is_valid_imgpath(self, imgpath):
        return '.JPG' in imgpath or '.PNG' in imgpath \
                or '.jpg' in imgpath or '.png' in imgpath \
                or '.ppm' in imgpath
    
    def setup_data_df(self):
        files_list_train = []
        files_list_test = []
        
        for dirpath in self.train_dirs_paths:
            if os.path.isdir(dirpath):
                for imgpath in os.listdir(dirpath):
                    if self.__is_valid_imgpath(imgpath):
                        files_list_train.append(os.path.join(dirpath, imgpath))
        
        for dirpath in self.test_dirs_paths:
            if os.path.isdir(dirpath):
                for imgpath in os.listdir(dirpath):
                    if self.__is_valid_imgpath(imgpath):
                        files_list_test.append(os.path.join(dirpath, imgpath))
        
        print('len(files_list_train): ', len(files_list_train))
        print('len(files_list_test): ', len(files_list_test))
        
        files_list_train = sorted(files_list_train)
        files_list_test = sorted(files_list_test)
        
        all_imgs = files_list_train + files_list_test
        
        print(f'total number of images: {len(all_imgs)}')
        
        if self.max_imgs is not None:
            np.random.seed(0)
            all_imgs = np.random.choice(all_imgs, self.max_imgs, replace=False)
            
        self.data_df = pd.DataFrame(columns=['img_path'])
        self.data_df['img_path'] = all_imgs
        self.data_df.to_csv(self.output_path, index=False)
        print('Images names dataframe saved')
        
