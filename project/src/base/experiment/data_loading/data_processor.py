import os
import pandas as pd

from neptune.new.types import File

from typing import List

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.base.experiment.dataset.dataset import Dataset
from src.base.experiment.evaluation.eval import Eval
from src.base.experiment.tasks.task import TASK
from src.base.net_data_loaders.net_gt_loader import NetGTLoader
from src.m_utils.constants import SEED, BASE_PATH
from src.base.gt_loaders.gt_names import GTName

class DataProcessor:
    def __init__(self, config_interp, neptune_utils):
        self.config_interp = config_interp
        self.neptune_run = neptune_utils.neptune_run
        
        self.train_data, self.validation_data, self.test_data = None, None, None


    def __load_gt_data(self):
        tasks: List[TASK] = self.config_interp.tasks
        aligned = False
        
        netGtLoader = NetGTLoader(aligned, 
                                  tasks, 
                                  [GTName.FVC], 
                                  self.config_interp.is_mtl_model)
            
        self.train_data = netGtLoader.load_gt_data(split='train')
        self.validation_data = netGtLoader.load_gt_data(split='validation')
        self.test_data = netGtLoader.load_gt_data(split='test')


    def __load_benchmark_data(self):
        self.train_data = pd.read_csv(os.path.join(BASE_PATH, self.config_interp.dataset.value['name'], 'train_data.csv'))
        print(f'TrainData.shape: {self.train_data.shape}')

        self.validation_data = pd.read_csv(os.path.join(BASE_PATH, self.config_interp.dataset.value['name'], 'valid_data.csv'))
        print(f'ValidationData.shape: {self.validation_data.shape}')

        self.test_data = pd.read_csv(os.path.join(BASE_PATH, self.config_interp.dataset.value['name'], 'test_data.csv'))
        print(f'TestData.shape: {self.test_data.shape}')


    def __transform_dtype_int2float(self):
        if self.config_interp.is_mtl_model:
            tasks = self.config_interp.tasks
            for task in tasks:
                self.train_data[task.value]      = self.train_data[task.value].astype(float)
                self.validation_data[task.value] = self.validation_data[task.value].astype(float)
                self.test_data[task.value]       = self.test_data[task.value].astype(float)

    
    def load_training_data(self):
        print('Loading data')

        if self.config_interp.dataset.name != Dataset.FVC_ICAO.name:
            self.__load_benchmark_data()
            self.__transform_dtype_int2float()
        else:
            self.__load_gt_data()
        
        print('Data loaded')  

    
    def sample_training_data(self, sample_prop):
        print('Applying subsampling in training data')
        total_train = self.train_data.shape[0]
        print(f"..Sampling proportion: {sample_prop} ({int(sample_prop * total_train)}/{total_train})")
        self.train_data = self.train_data.sample(frac=self.config_interp.prop_args['sample_prop'], random_state=SEED)
        print(self.train_data.shape)

        print('Applying subsampling in validation data')
        total_valid = self.validation_data.shape[0]
        print(f"..Sampling proportion: {sample_prop} ({int(sample_prop * total_valid)}/{total_valid})")
        self.validation_data = self.validation_data.sample(frac=self.config_interp.prop_args['sample_prop'], random_state=SEED)
        print(self.validation_data.shape)
    
    
    def balance_input_data(self, req_name):
        print(f'Requisite: {req_name}')
        
        print('Balancing input dataset..')
        final_df = pd.DataFrame()
        
        df_comp = self.train_data[self.train_data[req_name] == str(Eval.COMPLIANT.value)]
        df_non_comp = self.train_data[self.train_data[req_name] == str(Eval.NON_COMPLIANT.value)]

        print(f'df_comp.shape: {df_comp.shape}, df_non_comp.shape: {df_non_comp.shape}')

        n_imgs_non_comp, n_imgs_comp = df_non_comp.shape[0], df_comp.shape[0]

        final_df = pd.DataFrame()
        tmp_df = pd.DataFrame()
        if n_imgs_non_comp >= n_imgs_comp:
            print('n_imgs_non_comp >= n_imgs_comp')
            tmp_df = df_non_comp.sample(n_imgs_comp, random_state=SEED)
            final_df = final_df.append(df_comp)
            final_df = final_df.append(tmp_df)
        else:
            print('n_imgs_non_comp < n_imgs_comp')
            tmp_df = df_comp.sample(n_imgs_non_comp, random_state=SEED)
            final_df = final_df.append(df_non_comp)
            final_df = final_df.append(tmp_df) 

        print('final_df.shape: ', final_df.shape)
        print('n_comp: ', final_df[final_df[req_name] == str(Eval.COMPLIANT.value)].shape[0])
        print('n_non_comp: ', final_df[final_df[req_name] == str(Eval.NON_COMPLIANT.value)].shape[0])

        self.train_data = final_df
        print('Input dataset balanced')
        
    
    def __setup_fvc_class_mode(self):
        _class_mode, _y_col = None, None
        tasks = self.config_interp.tasks
        if self.config_interp.is_mtl_model:  
            _y_col = [t.value for t in tasks]
            _class_mode = 'multi_output'
        else: 
            _y_col = tasks[0].value
            _class_mode = 'categorical'
        return _class_mode,_y_col


    def __setup_benchmark_class_mode(self):
        _class_mode, _y_col = None, None
        if self.config_interp.is_mtl_model:  
            _y_col = [col for col in self.config_interp.dataset.value['target_cols']]
            _class_mode = 'multi_output'
        else:    
            raise NotImplemented()
        return _class_mode,_y_col


    def __setup_data_generators(self, base_model):
        train_datagen = None
        if self.config_interp.dataset.name == Dataset.FVC_ICAO.name:
            train_datagen = ImageDataGenerator(preprocessing_function=base_model.value['prep_function'], 
                                        horizontal_flip=True,
                                        #rotation_range=20,
                                        zoom_range=0.15,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.15,
                                        fill_mode="nearest")
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=base_model.value['prep_function'])
        
        validation_datagen = ImageDataGenerator(preprocessing_function=base_model.value['prep_function'])
        test_datagen = ImageDataGenerator(preprocessing_function=base_model.value['prep_function'])
        
        return train_datagen, validation_datagen, test_datagen


    def setup_data_generators(self, base_model):
        print('Starting data generators')
        
        train_datagen, validation_datagen, test_datagen,  = self.__setup_data_generators(base_model)

        if self.config_interp.dataset.name == Dataset.FVC_ICAO.name:    
            _class_mode, _y_col = self.__setup_fvc_class_mode()
        else:
            _class_mode, _y_col = self.__setup_benchmark_class_mode()


        self.train_gen = train_datagen.flow_from_dataframe(self.train_data, 
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.config_interp.mlp_params['mlp_batch_size'], 
                                                shuffle=True,
                                                seed=SEED)

        self.validation_gen = validation_datagen.flow_from_dataframe(self.validation_data,
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.config_interp.mlp_params['mlp_batch_size'],
                                                shuffle=False)

        self.test_gen = test_datagen.flow_from_dataframe(self.test_data,
                                               x_col="img_name", 
                                               y_col=_y_col,
                                               target_size=base_model.value['target_size'],
                                               class_mode=_class_mode,
                                               batch_size=self.config_interp.mlp_params['mlp_batch_size'],
                                               shuffle=False)

        print(f'TOTAL: {self.train_gen.n + self.validation_gen.n + self.test_gen.n}')
    
        self.__log_class_indices()
        self.__log_class_labels()

    
    def __log_class_indices(self):
        print('')
        print('Logging class indices')
        
        if not self.config_interp.is_mtl_model:
        
            train_class_indices = self.train_gen.class_indices
            valid_class_indices = self.validation_gen.class_indices
            test_class_indices = self.test_gen.class_indices

            print(f' ..Train Generator: {train_class_indices}')
            print(f' ..Valid Generator: {valid_class_indices}')
            print(f' ..Test Generator: {test_class_indices}')

            if self.config_interp.use_neptune:
                self.neptune_run['properties/class_indices_train'] = str(train_class_indices)
                self.neptune_run['properties/class_indices_valid'] = str(valid_class_indices)
                self.neptune_run['properties/class_indices_test'] = str(test_class_indices)
        
        else:
            print(' .. MTL model not logging class indices!')
    
    
    def __log_class_labels(self):
        print('')
        if self.config_interp.dataset.name == Dataset.FVC_ICAO.name:
            print('Logging class labels')
            
            print(f' COMPLIANT label: {Eval.COMPLIANT.value}')
            print(f' NON_COMPLIANT label: {Eval.NON_COMPLIANT.value}')
            print(f' DUMMY label: {Eval.DUMMY.value}')
            print(f' DUMMY_CLS label: {Eval.DUMMY_CLS.value}')
            print(f' NO_ANSWER label: {Eval.NO_ANSWER.value}')
            
            if self.config_interp.use_neptune:
                self.neptune_run['properties/labels'] = str({'compliant':Eval.COMPLIANT.value, 
                                                            'non_compliant':Eval.NON_COMPLIANT.value,
                                                            'dummy':Eval.DUMMY.value,
                                                            'dummy_cls':Eval.DUMMY_CLS.value,
                                                            'no_answer':Eval.NO_ANSWER.value})
        else:
            print('Using benchmarking dataset. Not logging class labels!')
    

    def get_summary(self, orig_df):
        df = pd.DataFrame()

        comp_val = Eval.COMPLIANT.value if self.config_interp.is_mtl_model else str(Eval.COMPLIANT.value)
        non_comp_val = Eval.NON_COMPLIANT.value if self.config_interp.is_mtl_model else str(Eval.NON_COMPLIANT.value)
        dummy_val = Eval.DUMMY_CLS.value if self.config_interp.is_mtl_model else str(Eval.DUMMY_CLS.value)

        for col in orig_df.columns:
            total = orig_df.shape[0]

            n_comp = orig_df[orig_df[col] == comp_val].shape[0]
            n_comp_perc = round(n_comp / total * 100, 2)
            
            n_non_comp = orig_df[orig_df[col] == non_comp_val].shape[0]
            n_non_comp_perc = round(n_non_comp / total * 100, 2)
            
            n_dummy = orig_df[orig_df[col] == dummy_val].shape[0]
            n_dummy_perc = round(n_dummy / total * 100, 2)

            total_comp_n_comp_dummy = n_comp + n_non_comp + n_dummy
            total_perc = n_comp_perc + n_non_comp_perc
            
            aux_df = pd.DataFrame.from_dict({'task':[col], 
                                            'n_comp':[n_comp], 
                                            'n_comp_perc':[n_comp_perc],
                                            'n_non_comp':[n_non_comp],
                                            'n_non_comp_perc':[n_non_comp_perc],
                                            'n_dummy':[n_dummy],
                                            'n_dummy_perc':[n_dummy_perc],
                                            'total_comp_n_comp_dummy':[total_comp_n_comp_dummy],
                                            'total_perc':[total_perc]})
            df = pd.concat([df,aux_df],ignore_index=True)
        
        df = df[df['task'].isin(x.value for x in self.config_interp.tasks)]

        return df

    
    def summary_labels_dist(self):
        data_list = [('train', self.train_data), ('validation', self.validation_data), ('test', self.test_data)]

        for data_name, data in data_list:
            data = data.iloc[:,2:]
            summary_data = self.get_summary(data)

            if self.config_interp.use_neptune:
                self.neptune_run[f'data_props/{data_name}/{data_name}_data_summary'].upload(File.as_html(summary_data))
        
    
    def summary_gen_labels_dist(self):
        data_list = [('train', self.train_gen), ('validation', self.validation_gen), ('test', self.test_gen)]

        if self.config_interp.dataset.name == Dataset.FVC_ICAO.name:    
            for data_name,data_gen in data_list:
                total = data_gen.n
                n_comp = len([x for x in data_gen.labels if x == Eval.COMPLIANT.value])
                n_non_comp = len([x for x in data_gen.labels if x == Eval.NON_COMPLIANT.value])
                n_dummy = len([x for x in data_gen.labels if x == Eval.DUMMY_CLS.value])
                
                prop_n_comp = round(n_comp/total*100,2)
                prop_n_non_comp = round(n_non_comp/total*100,2)
                prop_n_dummy =  round(n_dummy/total*100,2)     

                print(f'GEN_N_{data_name.upper()}_COMP: {n_comp} ({prop_n_comp}%)')
                print(f'GEN_N_{data_name.upper()}_NON_COMP: {n_non_comp} ({prop_n_non_comp}%)')
                print(f'GEN_N_{data_name.upper()}_DUMMY: {n_dummy} ({prop_n_dummy}%)')
                
                if self.config_interp.use_neptune:
                    neptune_vars_base_path = f'data_props/generators'
                    
                    self.neptune_run[f'{neptune_vars_base_path}/gen_total_{data_name}'] = total
                    self.neptune_run[f'{neptune_vars_base_path}/gen_n_{data_name}_comp'] = n_comp
                    self.neptune_run[f'{neptune_vars_base_path}/gen_n_{data_name}_non_comp'] = n_non_comp
                    self.neptune_run[f'{neptune_vars_base_path}/gen_n_{data_name}_dummy'] = n_dummy
                    self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_{data_name}_comp'] = prop_n_comp
                    self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_{data_name}_non_comp'] = prop_n_non_comp
                    self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_{data_name}_dummy'] = prop_n_dummy