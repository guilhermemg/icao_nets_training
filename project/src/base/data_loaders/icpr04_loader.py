import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class ICPR04_DL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.ICPR04, aligned, f'{cts.BASE_PATH}/icpr_04_database', True)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        path = None
        if self.aligned:
            path = os.path.join(self.dataset_path, 'AlignedFront/class_name')
        else:
            path = os.path.join(self.dataset_path, 'Front/deFace')
        
        self.train_dirs_paths = [path]
        self.test_dirs_paths = []
    
