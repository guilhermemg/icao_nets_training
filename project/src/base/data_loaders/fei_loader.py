from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class FeiDB_DL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.FEI_DB, aligned, f'{cts.BASE_PATH}/fei_database', True)
        
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        if self.aligned:
            self.train_dirs_paths = [f'{self.dataset_path}/aligned/']
            self.test_dirs_paths = [] 
        else:
            self.train_dirs_paths = [f'{self.dataset_path}/not_aligned/']
            self.test_dirs_paths = []