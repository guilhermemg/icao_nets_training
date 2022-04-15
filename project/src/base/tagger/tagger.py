import os
import pandas as pd

from src.m_utils.constants import BASE_OUTPUT_PATH


class Tagger:
    def __init__(self, dataloader, model, requisite):
        self.dataloader = dataloader
        self.requisite = requisite
        self.model = model
        self.labels_df = None
        
        self.aligned_out_dir = f'{BASE_OUTPUT_PATH}/{self.model.get_model_name().value}/labels/aligned/'
        self.not_aligned_out_dir = f'{BASE_OUTPUT_PATH}/{self.model.get_model_name().value}/labels/not_aligned/'
    
    def set_dataloader(self, dloader):
        self.dataloader = dloader
    
    def __get_tagger_output_path(self):
        icaoreq = self.requisite.value
        dl_name = self.dataloader.get_name().value
        
        if self.dataloader.is_aligned():
            return os.path.join(self.aligned_out_dir, f'{icaoreq}_{dl_name}.csv')
        else:
            return os.path.join(self.not_aligned_out_dir, f'{icaoreq}_{dl_name}.csv')
    
    def save_labels_df(self):
        if not os.path.exists(self.aligned_out_dir):
            os.makedirs(self.aligned_out_dir)
        if not os.path.exists(self.not_aligned_out_dir):
            os.makedirs(self.not_aligned_out_dir)
        
        path = self.__get_tagger_output_path()
        self.labels_df.to_csv(path, index=False)
    
    def load_labels_df(self):
        path = self.__get_tagger_output_path()
        self.labels_df = pd.read_csv(path)
    
    def get_labels_df(self):
        return self.labels_df
    
    def run(self, thresh):
        self.model.load_outdf(self.dataloader)
        self.labels_df = self.model.out_df.copy()
        for idx,row in self.labels_df.iterrows():
            self.labels_df.at[idx,'comp'] = self.model.check_compliance(row, thresh)
        self.labels_df = self.labels_df[['img_name','comp']]

        