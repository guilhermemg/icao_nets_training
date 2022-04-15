import pandas as pd

from src.m_utils.constants import ICAO_DATASET_PATH
from src.base.gt_loaders.gt_list import ALL_GT_LOADERS_LIST
from src.base.gt_loaders.gen_gt import Eval
from src.base.net_data_loaders.gen_net_data_loader import GenNetDataLoader


class NetGTLoader(GenNetDataLoader):
    def __init__(self, aligned, requisites, gt_names, is_mtl_model):
        super().__init__(aligned, requisites)
        
        self.gt_names = gt_names
        self.is_mtl_model = is_mtl_model
        self.gt_list = self.__select_gts()
    
    
    def __select_gts(self):
        init_gt_list = ALL_GT_LOADERS_LIST
        
        final_gt_list = []
        for gt in init_gt_list:
            if gt.get_name() in self.gt_names and gt.is_aligned() == self.aligned:
                final_gt_list.append(gt)
        return final_gt_list
    
    
    def load_gt_data(self, split, ignore_dummies=False):
        in_data = pd.DataFrame(columns=['origin','img_name'] + [req.value for req in self.requisites])
                
        for gt in self.gt_list:
            print(f'Loading GT {gt.get_name().value.upper()} - {split.upper()} split...')
            gt.load_data()
            
            if split == 'train':
                tmp_df = gt.train_df
            elif split == 'validation':
                tmp_df = gt.validation_df
            elif split == 'test':
                tmp_df = gt.test_df
            
            # sort columns
            tmp_df = tmp_df[['img_path'] + [req.value for req in self.requisites]].copy()
            tmp_df['img_path'] = tmp_df.img_path.str.replace('ICAO_DATASET_PATH', ICAO_DATASET_PATH)
            tmp_df.rename(columns={'img_path':'img_name'}, inplace=True)
            tmp_df['origin'] = gt.get_name().value
            tmp_df['aligned'] = gt.is_aligned()
            
            if ignore_dummies:  # ignore dummies and empty labels
                n_init = tmp_df.shape[0]
                for req in self.requisites:
                    tmp_df = tmp_df[(tmp_df[req.value] == Eval.COMPLIANT.value) | \
                                    (tmp_df[req.value] == Eval.NON_COMPLIANT.value)]  
                n_end = tmp_df.shape[0]
                print(f'..Ignoring {n_init - n_end} empty and dummies label values') 
            else:
                n_init = tmp_df.shape[0]
                for req in self.requisites:
                    tmp_df = tmp_df[(tmp_df[req.value] == Eval.COMPLIANT.value) | \
                                    (tmp_df[req.value] == Eval.NON_COMPLIANT.value) | \
                                    (tmp_df[req.value] == Eval.DUMMY.value)]  
                n_end = tmp_df.shape[0]
                print(f'..Ignoring {n_init - n_end} empty label values') 
            
            in_data = in_data.append(tmp_df, ignore_index=True)

        print(f'Input data.shape: {in_data.shape}')
        
        for req in self.requisites:
            if self.is_mtl_model:
                in_data[req.value] = in_data[req.value].astype(float)
                in_data[req.value] = in_data[req.value].apply(lambda x : x if x != -1 else 2)
            else:
                in_data[req.value] = in_data[req.value].astype(str)
            
        return in_data