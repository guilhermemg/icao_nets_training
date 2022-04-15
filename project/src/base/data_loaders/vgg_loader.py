import os

from src.m_utils import constants as cts
from src.base.data_loaders.data_loader import DataLoader, DLName

class VggFace2DL(DataLoader):
    def __init__(self, aligned, max_imgs=50000):
        super().__init__(DLName.VGGFACE2, aligned, f'{cts.BASE_PATH}/vggface2', restricted_env=False, max_imgs=max_imgs)
        self.set_dirs_paths()
        super().set_output_path()
        super().load_data_df()
        
    def set_dirs_paths(self):
        TRAIN_ALIGN_PATH = f'{self.dataset_path}/vggface2_train/train_align'
        TEST_ALIGN_PATH = f'{self.dataset_path}/vggface2_test/test_align'

        TRAIN_PATH = f'{self.dataset_path}/vggface2_train/train'
        TEST_PATH = f'{self.dataset_path}/vggface2_test/test'
        
        if self.aligned:
            self.train_dirs_paths = [os.path.join(TRAIN_ALIGN_PATH, x) for x in sorted(os.listdir(TRAIN_ALIGN_PATH))]
            self.test_dirs_paths = [os.path.join(TEST_ALIGN_PATH, x) for x in sorted(os.listdir(TEST_ALIGN_PATH))] 
        else:
            self.train_dirs_paths = [os.path.join(TRAIN_PATH, x) for x in sorted(os.listdir(TRAIN_PATH))]
            self.test_dirs_paths = [os.path.join(TEST_PATH, x) for x in sorted(os.listdir(TEST_PATH))]
