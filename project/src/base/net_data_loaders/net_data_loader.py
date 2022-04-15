import pandas as pd

from src.base.net_data_loaders.gen_net_data_loader import GenNetDataLoader
from src.base.data_loaders.dl_list import ALL_DATA_LOADERS_LIST
from src.base.tagger.tagger import Tagger
from src.base.gt_loaders.gen_gt import Eval

class NetDataLoader(GenNetDataLoader):
    def __init__(self, model, requisite, dl_names, aligned):
        super().__init__(aligned, requisite)
        
        self.model = model
        self.dl_names = dl_names
        self.dl_list = self.__select_data_loaders()
        

    def __select_data_loaders(self):
        init_dl_list = ALL_DATA_LOADERS_LIST
        
        final_dl_list = []
        for dl in init_dl_list:
            if dl.get_name() in self.dl_names and dl.is_aligned() == self.aligned:
                final_dl_list.append(dl)
        return final_dl_list
                                  
        
    def load_data(self, ignore_dummies=True):
        in_data = pd.DataFrame(columns=['origin','img_name','comp'])
        
        for dl in self.dl_list:
            t = Tagger(dl, self.model, self.requisite)
            t.load_labels_df()
            tmp_df = t.labels_df
            tmp_df['origin'] = dl.get_name().value
            tmp_df['aligned'] = dl.is_aligned()
            tmp_df['comp'] = tmp_df.comp.apply(lambda x : str(int(x)))
            
            if ignore_dummies:
                n_init = tmp_df.shape[0]
                tmp_df = tmp_df[(tmp_df.comp == str(Eval.COMPLIANT.value)) | \
                                (tmp_df.comp == str(Eval.NON_COMPLIANT.value))]  # ignore dummies and empty labels
                n_end = tmp_df.shape[0]
                print(f'..Ignoring {n_init - n_end} dummy and empty label values')
            
            in_data = in_data.append(tmp_df)

        print(f'Input data.shape: {in_data.shape}')
        
        return in_data
    
    