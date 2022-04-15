from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class IJBC_DL(DataLoader):
    def __init__(self, aligned, max_imgs=50000):
        super().__init__(DLName.IJBC, aligned, f'{cts.BASE_PATH}/ijb/IJB/IJB-C/images', False, max_imgs)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        if self.aligned:
            self.train_dirs_paths = [f'{self.dataset_path}/frames_aligned/class_name'] 
            self.test_dirs_paths = [f'{self.dataset_path}/img_aligned/class_name'] 
        else:
            self.train_dirs_paths = [f'{self.dataset_path}/frames'] 
            self.test_dirs_paths = [f'{self.dataset_path}/img/']
        
    