import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class UniEssexDL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.UNI_ESSEX, aligned, f'{cts.BASE_PATH}/uni_essex_uk', False)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        path = None
        if self.aligned:
            path = os.path.join(self.dataset_path, 'aligned_data')
        else:            
            path = os.path.join(self.dataset_path, 'data')

        self.train_dirs_paths = [os.path.join(path, x) for x in sorted(os.listdir(path))]
        self.test_dirs_paths = []
        
    
