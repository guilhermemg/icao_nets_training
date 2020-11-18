import os
import sys
import cv2
import random
import datetime
import neptune
import tempfile
import argparse
import pprint
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model

import tensorflow.keras.backend as K

from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as prep_input_inceptionv3
from tensorflow.keras.applications.vgg19 import preprocess_input as prep_input_vgg19
from tensorflow.keras.applications.vgg16 import preprocess_input as prep_input_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as prep_input_resnet50v2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adamax, Adadelta
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import RandomNormal

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from enum import Enum

if '../../../notebooks/' not in sys.path:
    sys.path.append('../../../notebooks/')

import utils.constants as cts
import utils.draw_utils as dr

from gt_loaders.gen_gt import Eval
from models.oface_mouth_model import OpenfaceMouth
from data_loaders.data_loader import DLName
from net_data_loaders.net_data_loader import NetDataLoader
from net_data_loaders.net_gt_loader import NetGTLoader


## restrict memory growth -------------------

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")

## restrict memory growth -------------------    

class BaseModel(Enum):
    MOBILENET_V2 = { 'target_size' : (224,224), 'prep_function': prep_input_mobilenetv2 }
    INCEPTION_V3 = { 'target_size' : (299,299), 'prep_function': prep_input_inceptionv3 }
    VGG19 =        { 'target_size' : (224,224), 'prep_function': prep_input_vgg19 }
    VGG16 =        { 'target_size' : (224,224), 'prep_function': prep_input_vgg16 }
    RESNET50_V2 =  { 'target_size' : (224,224), 'prep_function': prep_input_resnet50v2 }

class Optimizer(Enum):
    ADAM = 'Adam'
    ADAM_CUST = 'AdamCustomized'
    SGD = 'SGD'
    SGD_CUST = 'SGDCustomized'
    SGD_NESTEROV = 'SGDNesterov'
    ADAMAX = 'Adamax'
    ADAMAX_CUST = 'AdamaxCustomized'
    ADAGRAD = 'Adagrad'
    ADAGRAD_CUST = 'AdagradCustomized'
    ADADELTA = 'Adadelta'
    
class NetworkTrainer:
    def __init__(self, **kwargs):
        self.use_neptune = kwargs['use_neptune']
        print('-----')
        print('Use Neptune: ', self.use_neptune)
        print('-----')
        
        print('===================')
        print('Args: ')
        pprint.pprint(kwargs)
        print('===================')
        
        self.exp_args = kwargs['exp_params']
        self.prop_args = kwargs['properties']
        self.net_args = kwargs['net_train_params']
        
        self.base_model = self.net_args['base_model']
        print('----')
        print('Base Model Name: ', self.base_model)
        print('----')
        
        print('----')
        self.is_mtl_model = len(self.prop_args['reqs']) > 1
        print(f'MTL Model: {self.is_mtl_model}')
        print('----')
        
        self.CHECKPOINT_PATH = "training_ckpt/best_model.hdf5"
        self.__clear_checkpoints()

        
    def __clear_checkpoints(self):
        if os.path.exists(self.CHECKPOINT_PATH):
            os.remove(self.CHECKPOINT_PATH)
        
    
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
                                                 random_state=self.net_args['seed'])
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

    def balance_input_data(self):
        if self.prop_args['balance_input_data']:
            print('Balancing input dataset..')
            final_df = pd.DataFrame()

            df_comp = self.in_data[self.in_data.comp == Eval.COMPLIANT.value]
            df_non_comp = self.in_data[self.in_data.comp == Eval.NON_COMPLIANT.value]

            print(f'df_comp.shape: {df_comp.shape}, df_non_comp.shape: {df_non_comp.shape}')

            n_imgs = df_non_comp.shape[0]

            df_comp = df_comp[:n_imgs].copy()

            final_df = final_df.append(df_comp)
            final_df = final_df.append(df_non_comp)

            print('final_df.shape: ', final_df.shape)
            print('n_comp: ', final_df[final_df.comp == Eval.COMPLIANT.value].shape[0])
            print('n_non_comp: ', final_df[final_df.comp == Eval.NON_COMPLIANT.value].shape[0])

            self.in_data = final_df
            print('Input dataset balanced')
        else:
            print('Not balancing input_data')
        
    
    def start_neptune(self):
        if self.use_neptune:
            print('Starting Neptune')
            neptune.init('guilhermemg/icao-nets-training')    
        else:
            print('Not using Neptune')
    
       
    def setup_data_generators(self):
        print('Starting data generators')
        
        datagen = ImageDataGenerator(preprocessing_function=self.base_model.value['prep_function'], 
                                     validation_split=self.net_args['validation_split'],
                                     horizontal_flip=True,
                                     rotation_range=20,
                                     zoom_range=0.15,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.15,
                                     fill_mode="nearest")
        
        test_datagen = ImageDataGenerator(preprocessing_function=self.base_model.value['prep_function'])
        
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
                                                target_size=self.base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.net_args['batch_size'], 
                                                subset='training',
                                                shuffle=self.net_args['shuffle'],
                                                seed=self.net_args['seed'])

        self.validation_gen = datagen.flow_from_dataframe(self.train_data,
                                                x_col="img_name", 
                                                y_col=_y_col,
                                                target_size=self.base_model.value['target_size'],
                                                class_mode=_class_mode,
                                                batch_size=self.net_args['batch_size'], 
                                                subset='validation',
                                                shuffle=self.net_args['shuffle'],
                                                seed=self.net_args['seed'])

        self.test_gen = test_datagen.flow_from_dataframe(self.test_data,
                                               x_col="img_name", 
                                               y_col=_y_col,
                                               target_size=self.base_model.value['target_size'],
                                               class_mode=_class_mode,
                                               batch_size=self.net_args['batch_size'],
                                               shuffle=False)

        print(f'TOTAL: {self.train_gen.n + self.validation_gen.n + self.test_gen.n}')

    
    def summary_labels_dist(self):
        comp_val = Eval.COMPLIANT.value if self.is_mtl_model else str(Eval.COMPLIANT.value)
        non_comp_val = Eval.NON_COMPLIANT.value if self.is_mtl_model else str(Eval.NON_COMPLIANT.value)
        dummy_val = Eval.DUMMY.value if self.is_mtl_model else str(Eval.DUMMY.value)
        for req in self.prop_args['reqs']:
            print(f'Requisite: {req.value.upper()}')
            
            total_train_valid = self.train_data.shape[0]
            n_train_valid_comp = self.train_data[self.train_data[req.value] == comp_val].shape[0]
            n_train_valid_not_comp = self.train_data[self.train_data[req.value] == non_comp_val].shape[0]
            n_train_valid_dummy = self.train_data[self.train_data[req.value] == dummy_val].shape[0]

            total_test = self.test_data.shape[0]
            n_test_comp = self.test_data[self.test_data[req.value] == comp_val].shape[0]
            n_test_not_comp = self.test_data[self.test_data[req.value] == non_comp_val].shape[0]
            n_test_dummy = self.test_data[self.test_data[req.value] == dummy_val].shape[0]

            print(f'N_TRAIN_VALID_COMP: {n_train_valid_comp} ({round(n_train_valid_comp/total_train_valid*100,2)}%)')
            print(f'N_TRAIN_VALID_NOT_COMP: {n_train_valid_not_comp} ({round(n_train_valid_not_comp/total_train_valid*100,2)}%)')
            print(f'N_TRAIN_VALID_DUMMY: {n_train_valid_dummy} ({round(n_train_valid_dummy/total_train_valid*100,2)}%)')

            print(f'N_TEST_COMP: {n_test_comp} ({round(n_test_comp/total_test*100,2)}%)')
            print(f'N_TEST_NOT_COMP: {n_test_not_comp} ({round(n_test_not_comp/total_test*100,2)}%)')
            print(f'N_TEST_DUMMY: {n_test_dummy} ({round(n_test_dummy/total_test*100,2)}%)')
            
            print('----')
    

    def create_experiment(self):
        if self.use_neptune:
            print('Creating experiment')

            params = self.net_args
            params['n_train'] = self.train_gen.n
            params['n_validation'] = self.validation_gen.n
            params['n_test'] = self.test_gen.n

            props = {}
            if self.prop_args['use_gt_data']:
                props = {'gt_names': str(self.prop_args['gt_names'])}
            else:
                props = {
                    'dl_names': str([dl_n.value for dl_n in self.prop_args['dl_names']]),
                    'tagger_model': self.prop_args['tagger_model'].get_model_name().value
                }

            props['aligned'] = self.prop_args['aligned']
            props['icao_reqs'] = [r.value for r in self.prop_args['reqs']]
            props['balance_input_data'] = self.prop_args['balance_input_data']
            props['save_trained_model'] = self.prop_args['save_trained_model']
            props['is_mtl_model'] = self.is_mtl_model

            neptune.create_experiment(name=self.exp_args['name'],
                                      params=params,
                                      properties=props,
                                      description=self.exp_args['description'],
                                      tags=self.exp_args['tags'] ,
                                      upload_source_files=self.exp_args['src_files'])
        else:
            print('Not using Neptune')
    
    
    def __create_base_model(self):
        baseModel = None
        W,H = self.base_model.value['target_size']
        if self.base_model.name != BaseModel.INCEPTION_V3.name:
            if self.base_model.name == BaseModel.MOBILENET_V2.name:
                baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            elif self.base_model.name == BaseModel.VGG19.name:
                baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            elif self.base_model.name == BaseModel.VGG16.name:
                baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            elif self.base_model.name == BaseModel.RESNET50_V2.name:
                baseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        elif self.base_model.name == BaseModel.INCEPTION_V3.name:
            baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        return baseModel
    
        
    def __create_model(self):
        baseModel = self.__create_base_model()
        headModel = None
        if self.base_model.name != BaseModel.INCEPTION_V3.name:
            headModel = baseModel.output
        elif self.base_model.name == BaseModel.INCEPTION_V3.name:
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(8, 8))(headModel)

        initializer = RandomNormal(mean=0., stddev=1e-4, seed=self.net_args['seed'])
        
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu", kernel_initializer=initializer)(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(3, activation="softmax", kernel_initializer=initializer)(headModel)

        self.model = Model(inputs=baseModel.input, outputs=headModel)
        
        for layer in baseModel.layers:
            layer.trainable = False

        opt = self.__get_optimizer()

        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    
    def __create_mtl_model(self):
        baseModel = self.__create_base_model()
        
        for layer in baseModel.layers:
            layer.trainable = False
        
        initializer = RandomNormal(mean=0., stddev=1e-4, seed=self.net_args['seed'])
        
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)
        
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        
        y1 = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        y1 = Dropout(self.net_args['dropout'])(y1)
        y1 = Dense(64, activation='relu', kernel_initializer=initializer)(y1)
        y1 = Dropout(self.net_args['dropout'])(y1)
        
        y2 = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        y2 = Dropout(self.net_args['dropout'])(y2)
        y2 = Dense(64, activation='relu', kernel_initializer=initializer)(y2)
        y2 = Dropout(self.net_args['dropout'])(y2)
        
        y1 = Dense(3, activation='softmax', name='mouth', kernel_initializer=initializer)(y1)
        y2 = Dense(3, activation='softmax', name='veil', kernel_initializer=initializer)(y2)
        
        self.model = Model(inputs=baseModel.input, outputs=[y1,y2])
        
        opt = self.__get_optimizer()
        loss_list = ['sparse_categorical_crossentropy','sparse_categorical_crossentropy']
        metrics_list = ['accuracy']
        loss_weights = [.7,.2]
 
        self.model.compile(loss=loss_list, optimizer=opt, metrics=metrics_list)
    
    
    def create_model(self):
        print('Creating model...')
        if not self.is_mtl_model:
            self.__create_model()
        else:
            self.__create_mtl_model()

        if self.use_neptune:
            self.model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
        
        print('Model created')
    
    
    def vizualize_model(self):
        display(plot_model(self.model, show_shapes=True, to_file='figs/model.png'))
    
    
    def __get_optimizer(self):
        opt = None
        if self.net_args['optimizer'].name == Optimizer.ADAM.name:
            opt = Adam(lr=self.net_args['learning_rate'], decay=self.net_args['learning_rate'] / self.net_args['n_epochs'])
        elif self.net_args['optimizer'].name == Optimizer.ADAM_CUST.name:
            opt = Adam(lr=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.SGD.name:
            opt = SGD(lr=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.SGD_NESTEROV.name:
            opt = SGD(lr=self.net_args['learning_rate'], nesterov=True)
        elif self.net_args['optimizer'].name == Optimizer.ADAGRAD.name:
            opt = Adagrad(lr=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.ADAMAX.name:
            opt = Adamax(lr=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.ADADELTA.name:
            opt = Adadelta(lr=self.net_args['learning_rate'])
        return opt
    
    
    def __log_data(self, logs):
        neptune.log_metric('epoch_accuracy', logs['accuracy'])
        neptune.log_metric('epoch_val_accuracy', logs['val_accuracy'])
        neptune.log_metric('epoch_loss', logs['loss'])    
        neptune.log_metric('epoch_val_loss', logs['val_loss'])    

    def __lr_scheduler(self, epoch):
        if epoch <= 10:
            new_lr = self.net_args['learning_rate']
        elif epoch <= 20:
            new_lr = self.net_args['learning_rate'] * 0.5
        elif epoch <= 40:
            new_lr = self.net_args['learning_rate'] * 0.5
        else:
            new_lr = self.net_args['learning_rate'] * np.exp(0.1 * ((epoch//self.net_args['n_epochs'])*self.net_args['n_epochs'] - epoch))

        if self.use_neptune:
            neptune.log_metric('learning_rate', new_lr)
            
        return new_lr
    
    def __get_tensorboard_callback(self):
        log_dir = "tensorboard_out/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        return tensorboard_callback
    
    def __get_log_data_callback(self):
        return LambdaCallback(on_epoch_end = lambda epoch, logs: self.__log_data(logs))
    
    def __get_lr_scheduler_callback(self):
        return LearningRateScheduler(self.__lr_scheduler)
    
    def __get_early_stopping_callback(self):
        return EarlyStopping(patience=self.net_args['early_stopping'], 
                                         monitor='val_loss', 
                                         restore_best_weights=True)
    
    def __get_model_checkpoint_callback(self):
        return ModelCheckpoint("training_ckpt/best_model.hdf5", monitor='val_loss', save_best_only=True, mode='min')
    
    def __get_my_callback(self):
        class MyCallback(tf.keras.callbacks.Callback): 
            def __init__(self, val_gen):
                self.val_gen = val_gen
                self.out_file_path = 'output/out.csv'

            def __clean_out_file(self):
                if os.path.exists(self.out_file_path):
                    os.remove(self.out_file_path)

            def on_epoch_end(self, epoch, logs={}): 
                if epoch == 0:
                    self.__clean_out_file()

        #         print(f'Validation Accuracy: {self.model.evaluate(self.val_gen, batch_size=BS)}')

                with open(self.out_file_path,'a') as f:
                    predIxs = self.model.predict(self.val_gen, batch_size=BS)
                    Y = self.val_gen.labels
                    Y_hat = np.argmax(predIxs, axis=1)
                    for idx,(y,y_h) in enumerate(zip(Y,Y_hat)):
                        if epoch == 0 and idx == 0:
                            f.writelines('epoch,idx,y,y_hat\n')
                        f.writelines(f'{epoch+1},{idx},{y},{y_h}\n')   
        
        return MyCallback(self.validation_gen)


    def train_model(self):
        print(f'Training {self.base_model.name} network')

        callbacks_list = []
        
        if self.use_neptune:
            callbacks_list.append(self.__get_log_data_callback())

        if self.net_args['optimizer'].name in [Optimizer.ADAMAX_CUST.name, Optimizer.ADAGRAD_CUST.name,
                                               Optimizer.ADAM_CUST.name, Optimizer.SGD_CUST.name]:
            callbacks_list.append(self.__get_lr_scheduler_callback())
        
        if self.net_args['early_stopping'] is not None:
            callbacks_list.append(self.__get_early_stopping_callback())
        
        callbacks_list.append(self.__get_model_checkpoint_callback())
        
        self.H = self.model.fit(
                self.train_gen,
                steps_per_epoch=self.train_gen.n // self.net_args['batch_size'],
                validation_data=self.validation_gen,
                validation_steps=self.validation_gen.n // self.net_args['batch_size'],
                epochs=self.net_args['n_epochs'],
                callbacks=callbacks_list
        )
    
    
    def draw_training_history(self):
        f,ax = plt.subplots(1,2, figsize=(10,5))
        f.suptitle(f'-----{self.base_model.name}-----')

        ax[0].plot(self.H.history['accuracy'])
        ax[0].plot(self.H.history['val_accuracy'])
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].legend(['train', 'validation'])

        ax[1].plot(self.H.history['loss'])
        ax[1].plot(self.H.history['val_loss'])
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('loss')
        ax[1].set_xlabel('epoch')
        ax[1].legend(['train', 'validation'])

        plt.show()
        
        if self.use_neptune:
            neptune.send_image('training_curves.png',f)
    
    
    def load_checkpoint(self, chkp_name):
        self.__create_model()
        self.model.load_weights(chkp_name)
    

    def load_best_model(self):
        print('..Loading checkpoint')
        if os.path.isfile(self.CHECKPOINT_PATH):
            self.model.load_weights(self.CHECKPOINT_PATH)
            print('..Checkpoint weights loaded')
        else:
            print('Checkpoint not found')
    
    
    def save_model(self):
        if self.prop_args['save_trained_model']:
            print('Saving model')

            print('..Saving tf model')
            path = os.path.join('trained_models', 'model')
            self.model.save(path)
            print('..TF model saved')

            if self.use_neptune:
                print('..Saving model to neptune..')
                for item in os.listdir('trained_models'):
                    neptune.log_artifact(os.path.join('trained_models', item))

            self.model.training = False
        
            print('Model saved')
        else:
            print('Model not saved')


    def test_model(self):
        print("Testing Trained Model")
        self.test_gen.reset()
        predIdxs = self.model.predict(self.test_gen, batch_size=self.net_args['batch_size'])
        y_hat = np.argmax(predIdxs, axis=1)
        print(classification_report(y_true=self.test_gen.labels, y_pred=y_hat, target_names=['NON_COMP','COMP']))
        print(f'Model Accuracy: {round(accuracy_score(y_true=self.test_gen.labels, y_pred=y_hat), 4)}')      

    
    def evaluate_model(self, data_src='test'):
        print('Evaluating model')
        data = None
        if data_src == 'validation':
            data = self.validation_gen
        elif data_src == 'test':
            self.test_gen.reset()
            data = self.test_gen
            
        eval_metrics = self.model.evaluate(data, verbose=0)
        
        print(f'{data_src.upper()} loss: ', round(eval_metrics[0], 4))
        print(f'{data_src.upper()} accuracy: ', round(eval_metrics[1], 4))
        
        if self.use_neptune:
            for j, metric in enumerate(eval_metrics):
                neptune.log_metric('eval_' + self.model.metrics_names[j], metric)

                
    # Calculates heatmaps of GradCAM algorithm based on the following implementations:
    ## https://stackoverflow.com/questions/58322147/how-to-generate-cnn-heatmaps-using-built-in-keras-in-tf2-0-tf-keras 
    ## https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-gradcam-554a85dd4e48
    def __calc_heatmap(self, img_name):
        image = load_img(img_name, target_size=self.base_model.value['target_size'])
        img_tensor = img_to_array(image)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = self.base_model.value['prep_function'](img_tensor)

        last_conv_layer_name = [l.name for l in self.model.layers if isinstance(l, tf.python.keras.layers.convolutional.Conv2D)][-1]

        conv_layer = self.model.get_layer(last_conv_layer_name)
        heatmap_model = models.Model([self.model.inputs], [conv_layer.output, self.model.output])

        # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
        with tf.GradientTape() as gtape:
            conv_output, predictions = heatmap_model(img_tensor)
            loss = predictions[:, np.argmax(predictions[0])]
            grads = gtape.gradient(loss, conv_output)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # Channel-wise mean of resulting feature-map is the heatmap of class activation
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        # Render heatmap via pyplot
        # plt.imshow(heatmap[0])
        # plt.show()

        upsample = cv2.resize(heatmap[0], self.base_model.value['target_size'])
        return upsample
    
    # sort 50 samples from test_df, calculates GradCAM heatmaps
    # and log the resulting images in a grid to neptune
    def vizualize_predictions(self):
        predIdxs = self.model.predict(self.test_gen)
        preds = np.argmax(predIdxs, axis=1)  # NO SHUFFLE
        
        tmp_df = pd.DataFrame()
        tmp_df['img_name'] = self.test_gen.filepaths
        tmp_df['comp'] = self.test_gen.labels
        tmp_df['pred'] = preds
        
        tmp_df = tmp_df.sample(n = 50, random_state=42)
        
        def get_img_name(img_path):
            return img_path.split("/")[-1].split(".")[0]

        labels = [f'COMP\n {get_img_name(path)}' if x == Eval.COMPLIANT.value else f'NON_COMP\n {get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        preds = [f'COMP\n {get_img_name(path)}' if x == Eval.COMPLIANT.value else f'NON_COMP\n {get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        heatmaps = [self.__calc_heatmap(im_name) for im_name in tmp_df.img_name.values]
        
        imgs = [cv2.resize(cv2.imread(img),self.base_model.value['target_size']) for img in tmp_df.img_name.values]
        
        f = dr.draw_imgs(imgs, labels=labels, predictions=preds, heatmaps=heatmaps)
        
        if self.use_neptune:
            neptune.send_image('predictions_with_heatmaps.png',f)
    

    def finish_experiment(self):
        if self.use_neptune:
            print('Finishing Neptune')
            neptune.stop()
            self.use_neptune = False
        else:
            print('Not using Neptune')

        
    def run(self):
        self.load_training_data()
        self.balance_input_data()
        self.setup_data_generators()
        try:
            self.start_neptune()
            self.create_experiment()
            self.train_model()
            self.draw_training_history()
            self.save_model()
            self.test_model()
            self.evaluate_model()
            self.vizualize_predictions()
        except Exception as e:
            print(f'ERROR: {e}')
        finally:
            self.finish_experiment()
        