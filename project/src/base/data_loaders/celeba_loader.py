import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class CelebA_DL(DataLoader):
    def __init__(self, aligned, max_imgs=50000):
        super().__init__(DLName.CELEBA, aligned, f'{cts.BASE_PATH}/celebA/', restricted_env=False, max_imgs=max_imgs)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        path_list = []
        if self.aligned:
            path_list = [os.path.join(self.dataset_path,'img_align_celeba')]
        
        self.train_dirs_paths = path_list
        self.test_dirs_paths = []
    