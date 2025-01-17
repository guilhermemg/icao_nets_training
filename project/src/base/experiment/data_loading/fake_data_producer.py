import numpy as np
import pandas as pd


from src.m_utils import constants as cts


class FakeDataProducer:
    def __init__(self, config_interp, orig_train_data, orig_validation_data, orig_test_data):
        self.orig_train_data = orig_train_data
        self.orig_validation_data = orig_validation_data
        self.orig_test_data = orig_test_data
        self.config_interp = config_interp

        self.tasks = self.config_interp.tasks
        self.n_tasks = len(self.tasks)
    
        cols = self.orig_train_data.columns

        self.fake_train_data_df = pd.DataFrame(columns=cols)
        self.fake_validation_data_df = pd.DataFrame(columns=cols)
        self.fake_test_data_df = pd.DataFrame(columns=cols)
    
    
    def produce_data(self):
        rng = np.random.default_rng(cts.SEED)

        for i in range(500):
            d1 = {r.value: float(x) for r,x in zip(list(self.tasks), rng.integers(low=0, high=2, size=self.n_tasks))}
            d1['img_name'] = self.orig_train_data.img_name.values[i]
            #d1['origin'] = self.orig_train_data.origin.values[i]
            #d1['aligned'] = self.orig_train_data.aligned.values[i]
            self.fake_train_data_df = self.fake_train_data_df.append(d1, ignore_index=True)
            
        for i in range(100):
            d2 = {r.value: float(x) for r,x in zip(list(self.tasks), rng.integers(low=0, high=2, size=self.n_tasks))}
            d2['img_name'] = self.orig_validation_data.img_name.values[i]
            #d2['origin'] = self.orig_validation_data.origin.values[i]
            #d2['aligned'] = self.orig_validation_data.aligned.values[i]
            self.fake_validation_data_df = self.fake_validation_data_df.append(d2, ignore_index=True)

        for i in range(50):
            d3 = {r.value: float(x) for r,x in zip(list(self.tasks), rng.integers(low=0, high=2, size=self.n_tasks))}
            d3['img_name'] = self.orig_test_data.img_name.values[i]
            #d3['origin'] = self.orig_test_data.origin.values[i]
            #d3['aligned'] = self.orig_test_data.aligned.values[i]
            self.fake_test_data_df = self.fake_test_data_df.append(d3, ignore_index=True)
        
        print(f'fake_train_data.shape: {self.fake_train_data_df.shape}')
        print(f'fake_validation_data_df.shape: {self.fake_validation_data_df.shape}')
        print(f'fake_test_data_df.shape: {self.fake_test_data_df.shape}')