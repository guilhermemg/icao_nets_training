import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class CasiaWebface_DL(DataLoader):
    def __init__(self, aligned, max_imgs=50000):
        super().__init__(DLName.CASIA_WF, aligned, f'{cts.BASE_PATH}/casia_webface/', False, max_imgs)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        path_list = None
        if self.aligned:
            aligned_imgs_path = os.path.join(self.dataset_path,'aligned')
            path_list = [os.path.join(aligned_imgs_path,p) for p in sorted(os.listdir(aligned_imgs_path))]
        else:
            not_aligned_imgs_path = os.path.join(self.dataset_path,'not_aligned')
            path_list = [os.path.join(not_aligned_imgs_path,p) for p in sorted(os.listdir(not_aligned_imgs_path))]
        
        self.train_dirs_paths = path_list
        self.test_dirs_paths = []