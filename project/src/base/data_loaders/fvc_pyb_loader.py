from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class FvcPybossaDL(DataLoader):
    def __init__(self, aligned):
        super().__init__(DLName.FVC_PYBOSSA, aligned, cts.ICAO_DATASET_PATH, True)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        if self.aligned:
            self.train_dirs_paths = [f'{self.dataset_path}/train_align/class_name']
            self.test_dirs_paths = [f'{self.dataset_path}/validation_align/class_name'] 
        else:
            self.train_dirs_paths = [f'{self.dataset_path}/train/']
            self.test_dirs_paths = [f'{self.dataset_path}/validation/']
    
