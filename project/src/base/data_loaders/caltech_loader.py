import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class CaltechDL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.CALTECH, aligned, f'{cts.BASE_PATH}/caltech_frontal_faces', False)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        self.test_dirs_paths = []
        if self.aligned:
            self.train_dirs_paths = [os.path.join(self.dataset_path, 'aligned_faces/class_name')]
        else:
            self.train_dirs_paths = [os.path.join(self.dataset_path, 'faces')]
        
    
