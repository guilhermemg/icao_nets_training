import os
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.base.gt_loaders.gen_gt import Eval
from src.base.data_loaders.data_loader import DLName
from src.base.net_data_loaders.net_data_loader import NetDataLoader
from src.base.net_data_loaders.net_gt_loader import NetGTLoader

from src.m_utils.constants import SEED, BASE_PATH


class DataProcessor:
    def __init__(self, config_interp, neptune_utils):
        self.config_interp = config_interp
        self.neptune_run = neptune_utils.neptune_run
        
        self.train_data, self.validation_data, self.test_data = None, None, None


    def __load_gt_data(self):
        icao_data = self.config_interp.prop_args['icao_data']
        aligned, reqs, icao_gt = icao_data['aligned'], icao_data['reqs'], icao_data['icao_gt']
        is_many_datasets = len(icao_gt['gt_names']['train_validation_test']) == 0
        if is_many_datasets:
            trainNetGtLoader = NetGTLoader(aligned, 
                                           reqs, 
                                           icao_gt['gt_names']['train_validation'], 
                                           self.config_interp.is_mtl_model)
                
            self.train_data = trainNetGtLoader.load_gt_data(split='train')
            self.validation_data = trainNetGtLoader.load_gt_data(split='validation')
                
            print(f'TrainData.shape: {self.train_data.shape}')

            testNetGtLoader = NetGTLoader(aligned, 
                                          reqs, 
                                          icao_gt['gt_names']['test'], 
                                          self.config_interp.is_mtl_model)
                
            self.test_data = testNetGtLoader.load_gt_data(split='test')
                
            print(f'TestData.shape: {self.test_data.shape}')
                
        else:
            netGtLoader = NetGTLoader(aligned, 
                                      reqs, 
                                      icao_gt['gt_names']['train_validation_test'], 
                                      self.config_interp.is_mtl_model)
                
            self.train_data = netGtLoader.load_gt_data(split='train')
            self.validation_data = netGtLoader.load_gt_data(split='validation')
            self.test_data = netGtLoader.load_gt_data(split='test')
        
        #in_data = in_data.sample(frac=1.0, random_state=SEED)
        #np.random.seed(SEED)
        #train_prop = self.config_interp.net_args['train_prop']
        #valid_prop = self.config_interp.net_args['validation_prop']
        #self.train_data, self.validation_data, self.test_data = np.split(in_data, [int(train_prop*len(in_data)), 
        #                                                                           int((train_prop+valid_prop)*len(in_data))])
        
        #self.train_data = in_data.sample(frac=self.config_interp.net_args['train_prop']+self.config_interp.net_args['validation_prop'], random_state=SEED)
        #self.test_data = in_data[~in_data.img_name.isin(self.train_data.img_name)]


    def __load_dl_data(self):
        icao_data = self.config_interp.prop_args['icao_data']['icao_dl']
        netTrainDataLoader = NetDataLoader(icao_data['tagger_model'], 
                                           icao_data['reqs'], 
                                           icao_data['dl_names'], 
                                           icao_data['aligned'], 
                                           self.config_interp.is_mtl_model)
        self.train_data = netTrainDataLoader.load_data()
        print(f'TrainData.shape: {self.train_data.shape}')
            
        test_dataset = DLName.COLOR_FERET
        netTestDataLoader = NetDataLoader(icao_data['tagger_model'], 
                                          icao_data['reqs'], 
                                          [test_dataset], 
                                          icao_data['aligned'], 
                                          self.config_interp.is_mtl_model)
        self.test_data = netTestDataLoader.load_data()
        print(f'Test Dataset: {test_dataset.name.upper()}')
        print(f'TestData.shape: {self.test_data.shape}')


    def __load_benchmark_data(self):
        self.train_data = pd.read_csv(os.path.join(BASE_PATH, self.config_interp.benchmark_dataset.value['name'], 'train_data.csv'))
        print(f'TrainData.shape: {self.train_data.shape}')

        self.validation_data = pd.read_csv(os.path.join(BASE_PATH, self.config_interp.benchmark_dataset.value['name'], 'valid_data.csv'))
        print(f'ValidationData.shape: {self.validation_data.shape}')

        self.test_data = pd.read_csv(os.path.join(BASE_PATH, self.config_interp.benchmark_dataset.value['name'], 'test_data.csv'))
        print(f'TestData.shape: {self.test_data.shape}')

    
    def load_training_data(self):
        print('Loading data')

        if self.config_interp.use_benchmark_data:
            self.__load_benchmark_data()
        else:
            if self.config_interp.prop_args['icao_data']['icao_gt']['use_gt_data']:
                self.__load_gt_data()               
            else:
                self.__load_dl_data()
        
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
        reqs = self.config_interp.prop_args['icao_data']['reqs']
        if self.config_interp.is_mtl_model:  
            _y_col = [req.value for req in reqs]
            _class_mode = 'multi_output'
        else: 
            _y_col = reqs[0].value
            _class_mode = 'categorical'
        return _class_mode,_y_col


    def __setup_benchmark_class_mode(self):
        _class_mode, _y_col = None, None
        if self.config_interp.is_mtl_model:  
            _y_col = [col for col in self.config_interp.benchmark_dataset.value['target_cols']]
            _class_mode = 'multi_output'
        else:    
            raise NotImplemented()
        return _class_mode,_y_col


    def __setup_data_generators(self, base_model):
        train_datagen = None
        if not self.config_interp.use_benchmark_data:
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

        if not self.config_interp.use_benchmark_data:    
            _class_mode, _y_col = self.__setup_fvc_class_mode()
        else:
            _class_mode, _y_col = self.__setup_benchmark_class_mode()


        self.train_gen = train_datagen.flow_from_dataframe(self.train_data, 
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.config_interp.net_args['batch_size'], 
                                                shuffle=True,
                                                seed=SEED)

        self.validation_gen = validation_datagen.flow_from_dataframe(self.validation_data,
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.config_interp.net_args['batch_size'],
                                                shuffle=False)

        self.test_gen = test_datagen.flow_from_dataframe(self.test_data,
                                               x_col="img_name", 
                                               y_col=_y_col,
                                               target_size=base_model.value['target_size'],
                                               class_mode=_class_mode,
                                               batch_size=self.config_interp.net_args['batch_size'],
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
    
    
    def summary_labels_dist(self):
        comp_val = Eval.COMPLIANT.value if self.config_interp.is_mtl_model else str(Eval.COMPLIANT.value)
        non_comp_val = Eval.NON_COMPLIANT.value if self.config_interp.is_mtl_model else str(Eval.NON_COMPLIANT.value)
        dummy_val = Eval.DUMMY_CLS.value if self.config_interp.is_mtl_model else str(Eval.DUMMY_CLS.value)
        
        if not self.config_interp.use_benchmark_data:
            for req in self.config_interp.prop_args['icao_data']['reqs']:
                print(f'Requisite: {req.value.upper()}')
                
                total_train = self.train_data.shape[0]
                n_train_comp = self.train_data[self.train_data[req.value] == comp_val].shape[0]
                n_train_not_comp = self.train_data[self.train_data[req.value] == non_comp_val].shape[0]
                n_train_dummy = self.train_data[self.train_data[req.value] == dummy_val].shape[0]
                
                prop_n_train_comp = round((n_train_comp/total_train)*100,2)
                prop_n_train_not_comp = round((n_train_not_comp/total_train)*100,2)
                prop_n_train_dummy = round((n_train_dummy/total_train)*100,2)
                
                print(f'N_TRAIN_COMP: {n_train_comp} ({prop_n_train_comp}%)')
                print(f'N_TRAIN_NOT_COMP: {n_train_not_comp} ({prop_n_train_not_comp}%)')
                print(f'N_TRAIN_DUMMY: {n_train_dummy} ({prop_n_train_dummy}%)')
                
                total_validation = self.validation_data.shape[0]
                n_validation_comp = self.validation_data[self.validation_data[req.value] == comp_val].shape[0]
                n_validation_not_comp = self.validation_data[self.validation_data[req.value] == non_comp_val].shape[0]
                n_validation_dummy = self.validation_data[self.validation_data[req.value] == dummy_val].shape[0]

                prop_n_validation_comp = round(n_validation_comp/total_validation*100,2)
                prop_n_validation_not_comp = round(n_validation_not_comp/total_validation*100,2)
                prop_n_validation_dummy = round(n_validation_dummy/total_validation*100,2)
                
                print(f'N_VALIDATION_COMP: {n_validation_comp} ({prop_n_validation_comp}%)')
                print(f'N_VALIDATION_NOT_COMP: {n_validation_not_comp} ({prop_n_validation_not_comp}%)')
                print(f'N_VALIDATION_DUMMY: {n_validation_dummy} ({prop_n_validation_dummy}%)')
                
                total_test = self.test_data.shape[0]
                n_test_comp = self.test_data[self.test_data[req.value] == comp_val].shape[0]
                n_test_not_comp = self.test_data[self.test_data[req.value] == non_comp_val].shape[0]
                n_test_dummy = self.test_data[self.test_data[req.value] == dummy_val].shape[0]

                prop_n_test_comp = round(n_test_comp/total_test*100,2)
                prop_n_test_not_comp = round(n_test_not_comp/total_test*100,2)
                prop_n_test_dummy = round(n_test_dummy/total_test*100,2)
                
                print(f'N_TEST_COMP: {n_test_comp} ({prop_n_test_comp}%)')
                print(f'N_TEST_NOT_COMP: {n_test_not_comp} ({prop_n_test_not_comp}%)')
                print(f'N_TEST_DUMMY: {n_test_dummy} ({prop_n_test_dummy}%)')
                
                if self.config_interp.use_neptune:
                    neptune_vars_base_path = f'data_props/{req.value}'
                    
                    self.neptune_run[f'{neptune_vars_base_path}/total_train'] = total_train
                    self.neptune_run[f'{neptune_vars_base_path}/n_train_comp'] = n_train_comp
                    self.neptune_run[f'{neptune_vars_base_path}/n_train_not_comp'] = n_train_not_comp
                    self.neptune_run[f'{neptune_vars_base_path}/n_train_dummy'] = n_train_dummy
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_train_comp'] = prop_n_train_comp
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_train_not_comp'] = prop_n_train_not_comp
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_train_dummy'] = prop_n_train_dummy
                    
                    self.neptune_run[f'{neptune_vars_base_path}/total_validation'] = total_validation
                    self.neptune_run[f'{neptune_vars_base_path}/n_validation_comp'] = n_validation_comp
                    self.neptune_run[f'{neptune_vars_base_path}/n_validation_not_comp'] = n_validation_not_comp
                    self.neptune_run[f'{neptune_vars_base_path}/n_validation_dummy'] = n_validation_dummy
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_validation_comp'] = prop_n_validation_comp
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_validation_not_comp'] = prop_n_validation_not_comp
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_validation_dummy'] = prop_n_validation_dummy
                    
                    self.neptune_run[f'{neptune_vars_base_path}/total_test'] = total_test
                    self.neptune_run[f'{neptune_vars_base_path}/n_test_comp'] = n_test_comp
                    self.neptune_run[f'{neptune_vars_base_path}/n_test_not_comp'] = n_test_not_comp
                    self.neptune_run[f'{neptune_vars_base_path}/n_test_dummy'] = n_test_dummy
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_test_comp'] = prop_n_test_comp
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_test_not_comp'] = prop_n_test_not_comp
                    self.neptune_run[f'{neptune_vars_base_path}/prop_n_test_dummy'] = prop_n_test_dummy
                
                print('----')
        else:
            print('Using benchmark data. Not doing summary_labels_dist()')
    
    
    def summary_gen_labels_dist(self):
        if not self.config_interp.use_benchmark_data:    
            total_train = self.train_gen.n
            n_train_comp = len([x for x in self.train_gen.labels if x == Eval.COMPLIANT.value])
            n_train_non_comp = len([x for x in self.train_gen.labels if x == Eval.NON_COMPLIANT.value])
            n_train_dummy = len([x for x in self.train_gen.labels if x == Eval.DUMMY_CLS.value])
            
            prop_n_train_comp = round(n_train_comp/total_train*100,2)
            prop_n_train_non_comp = round(n_train_non_comp/total_train*100,2)
            prop_n_train_dummy =  round(n_train_dummy/total_train*100,2)     

            total_valid = self.validation_gen.n
            n_valid_comp = len([x for x in self.validation_gen.labels if x == Eval.COMPLIANT.value])
            n_valid_non_comp = len([x for x in self.validation_gen.labels if x == Eval.NON_COMPLIANT.value])
            n_valid_dummy = len([x for x in self.validation_gen.labels if x == Eval.DUMMY_CLS.value])
            
            prop_n_valid_comp = round(n_valid_comp/total_valid*100,2)
            prop_n_valid_non_comp = round(n_valid_non_comp/total_valid*100,2)
            prop_n_valid_dummy = round(n_valid_dummy/total_valid*100,2)
            
            total_test = self.test_gen.n
            n_test_comp= len([x for x in self.test_gen.labels if x == Eval.COMPLIANT.value])
            n_test_non_comp = len([x for x in self.test_gen.labels if x == Eval.NON_COMPLIANT.value])
            n_test_dummy = len([x for x in self.test_gen.labels if x == Eval.DUMMY_CLS.value])
            
            prop_n_test_comp = round(n_test_comp/total_test*100,2)
            prop_n_test_non_comp = round(n_test_non_comp/total_test*100,2)
            prop_n_test_dummy = round(n_test_dummy/total_test*100,2)

            print(f'GEN_N_TRAIN_COMP: {n_train_comp} ({prop_n_train_comp}%)')
            print(f'GEN_N_TRAIN_NON_COMP: {n_train_non_comp} ({prop_n_train_non_comp}%)')
            print(f'GEN_N_TRAIN_DUMMY: {n_train_dummy} ({prop_n_train_dummy}%)')
            
            print(f'GEN_N_VALID_COMP: {n_valid_comp} ({prop_n_valid_comp}%)')
            print(f'GEN_N_VALID_NON_COMP: {n_valid_non_comp} ({prop_n_valid_non_comp}%)')
            print(f'GEN_N_VALID_DUMMY: {n_valid_dummy} ({prop_n_valid_dummy}%)')

            print(f'GEN_N_TEST_COMP: {n_test_comp} ({prop_n_test_comp}%)')
            print(f'GEN_N_TEST_NON_COMP: {n_test_non_comp} ({prop_n_test_non_comp}%)')
            print(f'GEN_N_TEST_DUMMY: {n_test_dummy} ({prop_n_test_dummy}%)')
            
            if self.config_interp.use_neptune:
                neptune_vars_base_path = f'data_props/generators'
                
                self.neptune_run[f'{neptune_vars_base_path}/gen_total_train'] = total_train
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_train_comp'] = n_train_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_train_non_comp'] = n_train_non_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_train_dummy'] = n_train_dummy
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_train_comp'] = prop_n_train_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_train_non_comp'] = prop_n_train_non_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_train_dummy'] = prop_n_train_dummy
                
                self.neptune_run[f'{neptune_vars_base_path}/gen_total_valid'] = total_valid
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_valid_comp'] = n_valid_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_valid_non_comp'] = n_valid_non_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_valid_dummy'] = n_valid_dummy
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_valid_comp'] = prop_n_valid_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_valid_non_comp'] = prop_n_valid_non_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_valid_dummy'] = prop_n_valid_dummy
                
                self.neptune_run[f'{neptune_vars_base_path}/gen_total_test'] = total_test
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_test_comp'] = n_test_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_test_non_comp'] = n_test_non_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_n_test_dummy'] = n_test_dummy
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_test_comp'] = prop_n_test_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_test_non_comp'] = prop_n_test_non_comp
                self.neptune_run[f'{neptune_vars_base_path}/gen_prop_n_test_dummy'] = prop_n_test_dummy
        else:
            print('Using benchmark data. Not doing summary_gen_labels_dist()')