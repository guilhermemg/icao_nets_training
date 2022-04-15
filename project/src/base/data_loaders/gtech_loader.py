import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class GeorgiaTechDL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.GEORGIA_TECH, aligned, f'{cts.BASE_PATH}/georgiatech_db', False)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        train_dir_base_path = None
        if self.aligned:
            train_dir_base_path = f'{self.dataset_path}/GTdb_crop/'
        else:
            train_dir_base_path = f'{self.dataset_path}/gt_db/'
        
        self.train_dirs_paths = [os.path.join(train_dir_base_path, x) for x in sorted(os.listdir(train_dir_base_path))]
        self.test_dirs_paths = [] 
    
