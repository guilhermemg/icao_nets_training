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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if '../../../notebooks/' not in sys.path:
    sys.path.append('../../../notebooks/')

import utils.constants as cts
import utils.draw_utils as dr

from models.oface_mouth_model import OpenfaceMouth

from data_loaders.data_loader import DLName

from net_data_loaders.net_data_loader import NetDataLoader


## restrict memory growth -------------------

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")

## restrict memory growth -------------------    


class NetworkTrainer:
    def __init__(self, use_neptune=False, **kwargs):
        self.use_neptune = use_neptune
        
        print('===================')
        print('Args: ')
        pprint.pprint(kwargs)
        print('===================')
        
        self.exp_args = kwargs['exp_params']
        self.prop_args = kwargs['properties']
        self.net_args = kwargs['net_train_params']

    
    def load_training_data(self):
        print('Loading data')
        netDataLoader = NetDataLoader(self.prop_args['tagger_model'], self.prop_args['req'], 
                                      self.prop_args['dl_names'], self.prop_args['aligned'])
        self.in_data = netDataLoader.load_data()
        print(f'Number of Samples: {len(self.in_data)}')
        print('Data loaded')

    
    def start_neptune(self):
        print('Starting Neptune')
        neptune.init('guilhermemg/icao-nets-training')    
    
    
    def __log_data(self, logs):
        neptune.log_metric('epoch_accuracy', logs['accuracy'])
        neptune.log_metric('epoch_val_accuracy', logs['val_accuracy'])
        neptune.log_metric('epoch_loss', logs['loss'])    
        neptune.log_metric('epoch_val_loss', logs['val_loss'])    


    def __lr_scheduler(self, epoch):
        if epoch <= 10:
            new_lr = self.net_args['learning_rate']
#         elif epoch <= 20:
#             new_lr = self.net_args['learning_rate'] * 1e-1
#         elif epoch <= 40:
#             new_lr = self.net_args['learning_rate'] * 1e-2
        else:
            new_lr = self.net_args['learning_rate'] * np.exp(0.1 * ((epoch//100)*100 - epoch))
#             new_lr = self.net_args['learning_rate'] * 1e-3

        neptune.log_metric('learning_rate', new_lr)
        return new_lr


    def setup_data_generators(self):
        print('Starting data generators')
        train_prop,valid_prop = self.net_args['train_prop'], self.net_args['validation_prop']
        train_valid_df = self.in_data.sample(frac=train_prop+valid_prop, random_state=self.net_args['seed'])
        test_df = self.in_data[~self.in_data.img_name.isin(train_valid_df.img_name)]

        datagen = ImageDataGenerator(preprocessing_function=prep_input_mobilenetv2, 
                                     validation_split=self.net_args['validation_split'])

        self.train_gen = datagen.flow_from_dataframe(train_valid_df, 
                                                x_col="img_name", 
                                                y_col="comp",
                                                target_size=(224, 224),
                                                class_mode="binary",
                                                batch_size=self.net_args['batch_size'], 
                                                subset='training')

        self.validation_gen = datagen.flow_from_dataframe(train_valid_df,
                                                    x_col="img_name", 
                                                    y_col="comp",
                                                    target_size=(224, 224),
                                                    class_mode="binary",
                                                    batch_size=self.net_args['batch_size'], 
                                                    subset='validation')

        self.test_gen = datagen.flow_from_dataframe(test_df,
                                               x_col="img_name", 
                                               y_col="comp",
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               batch_size=self.net_args['batch_size'])

        print(f'TOTAL: {self.train_gen.n + self.validation_gen.n + self.test_gen.n}')


    def create_experiment(self):
        print('Creating experiment')
        
        params = self.net_args
        params['n_train'] = self.train_gen.n
        params['n_validation'] = self.validation_gen.n
        params['n_test'] = self.test_gen.n
        
        neptune.create_experiment(name=self.exp_args['name'],
                                  params=params,
                                  properties={'dl_names': str([dl_n.value for dl_n in self.prop_args['dl_names']]),
                                              'dl_aligned': self.prop_args['aligned'],
                                              'icao_req': self.prop_args['req'].value,
                                              'tagger_model': self.prop_args['tagger_model'].get_model_name().value},
                                  description=self.exp_args['description'],
                                  tags=self.exp_args['tags'] ,
                                  upload_source_files=self.exp_args['src_files'])
        
    def __create_model(self):
        baseModel = MobileNetV2(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)), input_shape=(224,224,3))

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(self.net_args['dense_units'], activation="relu")(headModel)
#         headModel = Dropout(self.net_args['dropout'])(headModel)
        headModel = Dense(1, activation="softmax")(headModel)

        self.model = Model(inputs=baseModel.input, outputs=headModel)
        
        for layer in baseModel.layers:
            layer.trainable = False
           
        #opt = Adam(lr=self.net_args['learning_rate'], decay=self.net_args['learning_rate'] / self.net_args['n_epochs'])
        self.model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
        
        
    def train_model(self):
        print('Training mobilenetv2 network')

        self.__create_model()

        # Log model summary
        self.model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="output/training/cp-{epoch:04d}.ckpt", verbose=1, 
                                                         save_weights_only=True, period=1)
        
        # train the head of the network
        H = self.model.fit(
                self.train_gen,
                steps_per_epoch=self.train_gen.n // self.net_args['batch_size'],
                validation_data=self.validation_gen,
                validation_steps=self.validation_gen.n // self.net_args['batch_size'],
                epochs=self.net_args['n_epochs'],
                callbacks=[LambdaCallback(on_epoch_end = lambda epoch, logs: self.__log_data(logs)),
                           EarlyStopping(patience=self.net_args['early_stopping'], monitor='accuracy', restore_best_weights=True),
                           LearningRateScheduler(self.__lr_scheduler),
                           cp_callback
                          ])
    
    
    def load_checkpoint(self, chkp_name):
        self.__create_model()
        self.model.load_weights(chkp_name)
    

    def save_model(self):
        print('Saving model')
        with tempfile.TemporaryDirectory(dir='.') as d:
            prefix = os.path.join(d, 'model_weights')
            self.model.save_weights(os.path.join(prefix, 'model.h5'))
            for item in os.listdir(prefix):
                neptune.log_artifact(os.path.join(prefix, item),
                                     os.path.join('model_weights', item))


    def test_model(self):
        print('Testing model')
        predIdxs = self.model.predict(self.test_gen, batch_size=self.net_args['batch_size'])
        predIdxs = np.argmax(predIdxs, axis=1)
        print(classification_report(self.test_gen.labels, predIdxs, target_names=['NON_COMP','COMP']))        

    
    def evaluate_model(self, data_src='test'):
        print('Evaluating model')
        data = None
        if data_src == 'validation':
            data = self.validation_gen
        elif data_src == 'test':
            data = self.test_gen
            
        eval_metrics = self.model.evaluate(data, verbose=0)
        
        print(f'{data_src.upper()} loss: ', eval_metrics[0])
        print(f'{data_src.upper()} accuracy: ', eval_metrics[1])
        
        if self.use_neptune:
            for j, metric in enumerate(eval_metrics):
                neptune.log_metric('eval_' + self.model.metrics_names[j], metric)


    def finish_experiment(self):
        print('Finishing Neptune')
        neptune.stop()

        
    def run(self):
        self.load_training_data()
        self.setup_data_generators()
        try:
            self.start_neptune()
            self.create_experiment()
            self.train_model()
            self.save_model()
            self.test_model()
            self.evaluate_model()
        except Exception as e:
            print(f'ERROR: {e}')
        finally:
            self.finish_experiment()
        