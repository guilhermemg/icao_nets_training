import neptune
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from gt_loaders.gen_gt import Eval
from data_loaders.data_loader import DLName
from net_data_loaders.net_data_loader import NetDataLoader
from net_data_loaders.net_gt_loader import NetGTLoader

from utils.constants import SEED


class DataProcessor:
    def __init__(self, prop_args, net_args, is_mtl_model, use_neptune):
        self.prop_args = prop_args
        self.net_args = net_args
        self.is_mtl_model = is_mtl_model
        self.use_neptune = use_neptune
        self.train_data, self.test_data = None, None
    
    
    def load_training_data(self):
        print('Loading data')
        
        if self.prop_args['use_gt_data']:
            if len(self.prop_args['gt_names']['train_validation_test']) == 0:
                trainNetGtLoader = NetGTLoader(self.prop_args['aligned'], self.prop_args['reqs'], 
                                               self.prop_args['gt_names']['train_validation'], self.is_mtl_model)
                self.train_data = trainNetGtLoader.load_gt_data()
                print(f'TrainData.shape: {self.train_data.shape}')

                testNetGtLoader = NetGTLoader(self.prop_args['aligned'], self.prop_args['reqs'], 
                                               self.prop_args['gt_names']['test'], self.is_mtl_model)
                self.test_data = testNetGtLoader.load_gt_data()
                print(f'TestData.shape: {self.test_data.shape}')
                
            else:
                netGtLoader = NetGTLoader(self.prop_args['aligned'], self.prop_args['reqs'], 
                                          self.prop_args['gt_names']['train_validation_test'], self.is_mtl_model)
                in_data = netGtLoader.load_gt_data()
                
                self.train_data = in_data.sample(frac=self.net_args['train_prop']+self.net_args['validation_prop'],
                                                 random_state=SEED)
                self.test_data = in_data[~in_data.img_name.isin(self.train_data.img_name)]
        else:
            netTrainDataLoader = NetDataLoader(self.prop_args['tagger_model'], self.prop_args['reqs'], 
                                          self.prop_args['dl_names'], self.prop_args['aligned'], self.is_mtl_model)
            self.train_data = netTrainDataLoader.load_data()
            print(f'TrainData.shape: {self.train_data.shape}')
            
            test_dataset = DLName.COLOR_FERET
            netTestDataLoader = NetDataLoader(self.prop_args['tagger_model'], self.prop_args['reqs'], 
                                          [test_dataset], self.prop_args['aligned'], self.is_mtl_model)
            self.test_data = netTestDataLoader.load_data()
            print(f'Test Dataset: {test_dataset.name.upper()}')
            print(f'TestData.shape: {self.test_data.shape}')
        
        print('Data loaded')

    
    def sample_training_data(self, sample_prop):
        print('Applying subsampling in training data')
        total_train_valid = self.train_data.shape[0]
        print(f"..Sampling proportion: {sample_prop} ({int(sample_prop * total_train_valid)}/{total_train_valid})")
        self.train_data = self.train_data.sample(frac=self.prop_args['sample_prop'], random_state=SEED)
        print(self.train_data.shape)
    
    
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
        
    
    
    def setup_data_generators(self, base_model):
        print('Starting data generators')
        
        datagen = ImageDataGenerator(preprocessing_function=base_model.value['prep_function'], 
                                     validation_split=self.net_args['validation_split'],
                                     horizontal_flip=True,
                                     rotation_range=20,
                                     zoom_range=0.15,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.15,
                                     fill_mode="nearest")
        
        test_datagen = ImageDataGenerator(preprocessing_function=base_model.value['prep_function'])
        
        _class_mode, _y_col = None, None
        if self.is_mtl_model:  
            _y_col = [req.value for req in self.prop_args['reqs']]
            _class_mode = 'multi_output'
        else:    
            _y_col = self.prop_args['reqs'][0].value
            _class_mode = 'categorical'
        
        self.train_gen = datagen.flow_from_dataframe(self.train_data, 
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.net_args['batch_size'], 
                                                subset='training',
                                                shuffle=True,
                                                seed=SEED)

        self.validation_gen = datagen.flow_from_dataframe(self.train_data,
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.net_args['batch_size'], 
                                                subset='validation',
                                                shuffle=True,
                                                seed=SEED)

        self.test_gen = test_datagen.flow_from_dataframe(self.test_data,
                                               x_col="img_name", 
                                               y_col=_y_col,
                                               target_size=base_model.value['target_size'],
                                               class_mode=_class_mode,
                                               batch_size=self.net_args['batch_size'],
                                               shuffle=False)

        print(f'TOTAL: {self.train_gen.n + self.validation_gen.n + self.test_gen.n}')
    
    
    def summary_labels_dist(self):
        comp_val = Eval.COMPLIANT.value if self.is_mtl_model else str(Eval.COMPLIANT.value)
        non_comp_val = Eval.NON_COMPLIANT.value if self.is_mtl_model else str(Eval.NON_COMPLIANT.value)
        #dummy_val = Eval.DUMMY.value if self.is_mtl_model else str(Eval.DUMMY.value)
        dummy_val = 2
        for req in self.prop_args['reqs']:
            print(f'Requisite: {req.value.upper()}')
            
            total_train_valid = self.train_data.shape[0]
            n_train_valid_comp = self.train_data[self.train_data[req.value] == comp_val].shape[0]
            n_train_valid_not_comp = self.train_data[self.train_data[req.value] == non_comp_val].shape[0]
            n_train_valid_dummy = self.train_data[self.train_data[req.value] == dummy_val].shape[0]

            prop_n_train_valid_comp = round(n_train_valid_comp/total_train_valid*100,2)
            prop_n_train_valid_not_comp = round(n_train_valid_not_comp/total_train_valid*100,2)
            prop_n_train_valid_dummy = round(n_train_valid_dummy/total_train_valid*100,2)
            
            print(f'N_TRAIN_VALID_COMP: {n_train_valid_comp} ({prop_n_train_valid_comp}%)')
            print(f'N_TRAIN_VALID_NOT_COMP: {n_train_valid_not_comp} ({prop_n_train_valid_not_comp}%)')
            print(f'N_TRAIN_VALID_DUMMY: {n_train_valid_dummy} ({prop_n_train_valid_dummy}%)')
            
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
            
            if self.use_neptune:
                neptune.log_metric(req.name + '_total_train_valid', total_train_valid)
                neptune.log_metric(req.name + '_n_train_valid_comp', n_train_valid_comp)
                neptune.log_metric(req.name + '_n_train_valid_not_comp', n_train_valid_not_comp)
                neptune.log_metric(req.name + '_n_train_valid_dummy', n_train_valid_dummy)
                neptune.log_metric(req.name + '_prop_n_train_valid_comp', prop_n_train_valid_comp)
                neptune.log_metric(req.name + '_prop_n_train_valid_not_comp', prop_n_train_valid_not_comp)
                neptune.log_metric(req.name + '_prop_n_train_valid_dummy', prop_n_train_valid_dummy)
                
                neptune.log_metric(req.name + '_total_test', total_test)
                neptune.log_metric(req.name + '_n_test_comp', n_test_comp)
                neptune.log_metric(req.name + '_n_test_not_comp', n_test_not_comp)
                neptune.log_metric(req.name + '_n_test_dummy', n_test_dummy)
                neptune.log_metric(req.name + '_prop_n_test_comp', prop_n_test_comp)
                neptune.log_metric(req.name + 'prop_n_test_not_comp', prop_n_test_not_comp)
                neptune.log_metric(req.name + '_prop_n_test_dummy', prop_n_test_dummy)
            
            print('----')
    
    
    def summary_gen_labels_dist(self):
        total_train = self.train_gen.n
        n_train_comp = len([x for x in self.train_gen.labels if x == 0])
        n_train_non_comp = len([x for x in self.train_gen.labels if x == 1])
        n_train_dummy = len([x for x in self.train_gen.labels if x == 2])
        
        prop_n_train_comp = round(n_train_comp/total_train*100,2)
        prop_n_train_non_comp = round(n_train_non_comp/total_train*100,2)
        prop_n_train_dummy =  round(n_train_dummy/total_train*100,2)     

        total_valid = self.validation_gen.n
        n_valid_comp = len([x for x in self.validation_gen.labels if x == 0])
        n_valid_non_comp = len([x for x in self.validation_gen.labels if x == 1])
        n_valid_dummy = len([x for x in self.validation_gen.labels if x == 2])
        
        prop_n_valid_comp = round(n_valid_comp/total_valid*100,2)
        prop_n_valid_non_comp = round(n_valid_non_comp/total_valid*100,2)
        prop_n_valid_dummy = round(n_valid_dummy/total_valid*100,2)
        
        total_test = self.test_gen.n
        n_test_comp= len([x for x in self.test_gen.labels if x == 0])
        n_test_non_comp = len([x for x in self.test_gen.labels if x == 1])
        n_test_dummy = len([x for x in self.test_gen.labels if x == 2])
        
        prop_n_test_comp = round(n_test_comp/total_test*100,2)
        prop_n_test_non_comp = round(n_test_non_comp/total_test*100,2)
        prop_n_test_dummy = round(n_test_dummy/total_test*100,2)

        print(f'N_TRAIN_COMP: {n_train_comp} ({prop_n_train_comp}%)')
        print(f'N_TRAIN_NON_COMP: {n_train_comp} ({prop_n_train_non_comp}%)')
        print(f'N_TRAIN_DUMMY: {n_train_dummy} ({prop_n_train_dummy}%)')
        
        print(f'N_VALID_COMP: {n_valid_comp} ({prop_n_valid_comp}%)')
        print(f'N_VALID_NON_COMP: {n_valid_non_comp} ({prop_n_valid_non_comp}%)')
        print(f'N_VALID_DUMMY: {n_valid_dummy} ({prop_n_valid_dummy}%)')

        print(f'N_TEST_COMP: {n_test_comp} ({prop_n_test_comp}%)')
        print(f'N_TEST_NON_COMP: {n_test_non_comp} ({prop_n_test_non_comp}%)')
        print(f'N_TEST_DUMMY: {n_test_dummy} ({prop_n_test_dummy}%)')
        
        if self.use_neptune:
            neptune.log_metric('gen_total_train', total_train)
            neptune.log_metric('gen_n_train_comp', n_train_comp)
            neptune.log_metric('gen_n_train_non_comp', n_train_non_comp)
            neptune.log_metric('gen_prop_n_train_comp', prop_n_train_comp)
            neptune.log_metric('gen_prop_n_train_non_comp', prop_n_train_non_comp)
            neptune.log_metric('gen_prop_n_train_dummy', prop_n_train_dummy)
            
            neptune.log_metric('gen_total_valid', total_valid)
            neptune.log_metric('gen_n_valid_comp', n_valid_comp)
            neptune.log_metric('gen_n_valid_non_comp', n_valid_non_comp)
            neptune.log_metric('gen_n_valid_dummy', n_valid_dummy)
            neptune.log_metric('gen_prop_n_valid_comp', prop_n_valid_comp)
            neptune.log_metric('gen_prop_n_valid_non_comp', prop_n_valid_non_comp)
            neptune.log_metric('gen_prop_n_valid_dummy', prop_n_valid_dummy)
            
            neptune.log_metric('gen_total_test', total_test)
            neptune.log_metric('gen_n_test_comp', n_test_comp)
            neptune.log_metric('gen_n_test_non_comp', n_test_non_comp)
            neptune.log_metric('gen_n_test_dummy', n_test_dummy)
            neptune.log_metric('gen_prop_n_test_comp', prop_n_test_comp)
            neptune.log_metric('gen_prop_n_test_non_comp', prop_n_test_non_comp)
            neptune.log_metric('gen_prop_n_test_dummy', prop_n_test_dummy)
            