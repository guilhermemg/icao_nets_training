from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class IMFDB_DL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.IMFDB, aligned, f'{cts.BASE_PATH}/indian_facial_db/database/Version 1/Images', True)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        self.train_dirs_paths = [self.dataset_path]
        self.test_dirs_paths = []
    
