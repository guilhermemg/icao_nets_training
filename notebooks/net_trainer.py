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

import tensorflow.keras.backend as K

from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as prep_input_inceptionv3
from tensorflow.keras.applications.vgg19 import preprocess_input as prep_input_vgg19
from tensorflow.keras.applications.resnet_v2 import preprocess_input as prep_input_resnet50v2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

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
    RESNET50_V2 =  { 'target_size' : (224,224), 'prep_function': prep_input_resnet50v2 }

class NetworkTrainer:
    def __init__(self, base_model, use_neptune=False, **kwargs):
        self.use_neptune = use_neptune
        self.base_model = base_model
        
        print('-----')
        print('Use Neptune: ', self.use_neptune)
        print('Base Model Name: ', self.base_model)
        print('-----')
        
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

    def balance_input_data(self):
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
        
    
    def start_neptune(self):
        print('Starting Neptune')
        neptune.init('guilhermemg/icao-nets-training')    
    
    
    def __log_data(self, logs):
        neptune.log_metric('epoch_accuracy', logs['accuracy'])
        neptune.log_metric('epoch_val_accuracy', logs['val_accuracy'])
        neptune.log_metric('epoch_loss', logs['loss'])    
        neptune.log_metric('epoch_val_loss', logs['val_loss'])    


    def __lr_scheduler(self, epoch):
#         if epoch <= 10:
#             new_lr = self.net_args['learning_rate']
#         elif epoch <= 20:
#             new_lr = self.net_args['learning_rate'] * 1e-1
#         elif epoch <= 40:
#             new_lr = self.net_args['learning_rate'] * 1e-2
#         else:
        new_lr = self.net_args['learning_rate'] * np.exp(0.1 * ((epoch//self.net_args['n_epochs'])*self.net_args['n_epochs'] - epoch))
#             new_lr = self.net_args['learning_rate'] * 1e-3

        if self.use_neptune:
            neptune.log_metric('learning_rate', new_lr)
            
        return new_lr

    
    def setup_data_generators(self):
        print('Starting data generators')
        train_prop,valid_prop = self.net_args['train_prop'], self.net_args['validation_prop']
        self.train_valid_df = self.in_data.sample(frac=train_prop+valid_prop, random_state=self.net_args['seed'])
        self.test_df = self.in_data[~self.in_data.img_name.isin(self.train_valid_df.img_name)]

        datagen = datagen = ImageDataGenerator(preprocessing_function=self.base_model.value['prep_function'], 
                                     validation_split=self.net_args['validation_split'])       

        self.train_gen = datagen.flow_from_dataframe(self.train_valid_df, 
                                                x_col="img_name", 
                                                y_col="comp",
                                                target_size=self.base_model.value['target_size'],
                                                class_mode="raw",
                                                batch_size=self.net_args['batch_size'], 
                                                subset='training',
                                                shuffle=self.net_args['shuffle'],
                                                seed=self.net_args['seed'])

        self.validation_gen = datagen.flow_from_dataframe(self.train_valid_df,
                                                x_col="img_name", 
                                                y_col="comp",
                                                target_size=self.base_model.value['target_size'],
                                                class_mode="raw",
                                                batch_size=self.net_args['batch_size'], 
                                                subset='validation',
                                                shuffle=self.net_args['shuffle'],
                                                seed=self.net_args['seed'])

        self.test_gen = datagen.flow_from_dataframe(self.test_df,
                                               x_col="img_name", 
                                               y_col="comp",
                                               target_size=self.base_model.value['target_size'],
                                               class_mode="raw",
                                               batch_size=self.net_args['batch_size'],
                                               shuffle=self.net_args['shuffle'],
                                               seed=self.net_args['seed'])

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
        baseModel, headModel = None, None
        
        W,H = self.base_model.value['target_size']
        if self.base_model.name != BaseModel.INCEPTION_V3.name:
            if self.base_model.name == BaseModel.MOBILENET_V2.name:
                baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            elif self.base_model.name == BaseModel.VGG19.name:
                baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            elif self.base_model.name == BaseModel.RESNET50_V2.name:
                baseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        elif self.base_model.name == BaseModel.INCEPTION_V3.name:
            baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(8, 8))(headModel)
            
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(self.net_args['dense_units'], activation="relu")(headModel)
        headModel = Dropout(self.net_args['dropout'])(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        self.model = Model(inputs=baseModel.input, outputs=headModel)
        
        for layer in baseModel.layers:
            layer.trainable = False
           
        opt = Adam(lr=self.net_args['learning_rate'], decay=self.net_args['learning_rate'] / self.net_args['n_epochs'])
        self.model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        
    
    def __get_tensorboard_callback(self):
        log_dir = "tensorboard_out/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        return tensorboard_callback
    
    def __get_model_checkpoint_callback(self):
        checkpoint_filepath = '/output/checkpoint_epoch_{}'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_freq='epoch')
        return model_checkpoint_callback
    
    def __get_log_data_callback(self):
        return LambdaCallback(on_epoch_end = lambda epoch, logs: self.__log_data(logs))
    
    def __get_lr_scheduler_callback(self):
        return LearningRateScheduler(self.__lr_scheduler)
    
    def __get_early_stopping_callback(self):
        return EarlyStopping(patience=self.net_args['early_stopping'], 
                                         monitor='accuracy', 
                                         restore_best_weights=True)
    
    def train_model(self):
        print(f'Training {self.base_model.name} network')

        self.__create_model()

        # Log model summary
        if self.use_neptune:
            self.model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
        
        callbacks_list = []
        if self.use_neptune:
            callbacks_list.append(self.__get_log_data_callback())
        callbacks_list.append(self.__get_lr_scheduler_callback())
        callbacks_list.append(self.__get_early_stopping_callback())
        
        # train the head of the network
        self.H = self.model.fit(
                self.train_gen,
                steps_per_epoch=self.train_gen.n // self.net_args['batch_size'],
                validation_data=self.validation_gen,
                validation_steps=self.validation_gen.n // self.net_args['batch_size'],
                epochs=self.net_args['n_epochs'],
                callbacks=callbacks_list)
    
    
    def draw_training_history(self):
        f,ax = plt.subplots(1,2, figsize=(10,5))
        f.suptitle(f'-----{self.base_model.name}-----')

        ax[0].plot(self.H.history['accuracy'])
        ax[0].plot(self.H.history['val_accuracy'])
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].legend(['train', 'test'])

        ax[1].plot(self.H.history['loss'])
        ax[1].plot(self.H.history['val_loss'])
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('loss')
        ax[1].set_xlabel('epoch')
        ax[1].legend(['train', 'test'])

        plt.show()
    
    
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
        print("Testing Trained Model")
        predIdxs = self.model.predict(self.test_gen, batch_size=self.net_args['batch_size'])
        y_hat = np.argmax(predIdxs, axis=1)
        print(classification_report(self.test_gen.labels, y_hat, target_names=['NON_COMP','COMP']))
        print(f'Model Accuracy: {round(accuracy_score(self.test_gen.labels, y_hat), 4)}')      

    
    def evaluate_model(self, data_src='test'):
        print('Evaluating model')
        data = None
        if data_src == 'validation':
            data = self.validation_gen
        elif data_src == 'test':
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
        preds = np.argmax(self.model.predict(self.test_gen), axis=1)
        cnt = 0
        for idx,_ in self.test_df.iterrows():
            self.test_df.loc[idx, 'pred'] = preds[cnt]
            cnt += 1
        
        tmp_df = self.test_df.sample(n = 50, random_state=42)

        def get_img_name(img_path):
            return img_path.split("/")[-1].split(".")[0]

        labels = [f'COMP\n {get_img_name(path)}' if x == Eval.COMPLIANT.value else f'NON_COMP\n {get_img_name(path)}' for x,path in zip(tmp_df.comp.values, tmp_df.img_name.values)]
        preds = [f'COMP\n {get_img_name(path)}' if x == Eval.COMPLIANT.value else f'NON_COMP\n {get_img_name(path)}' for x,path in zip(tmp_df.pred.values, tmp_df.img_name.values)]
        heatmaps = [self.__calc_heatmap(im_name) for im_name in tmp_df.img_name.values]

        f = dr.draw_imgs([cv2.imread(img) for img in tmp_df.img_name.values], labels=labels, predictions=preds, heatmaps=heatmaps)
        
        if self.use_neptune:
            neptune.send_image('predictions_with_heatmaps.png',f)
    

    def finish_experiment(self):
        print('Finishing Neptune')
        neptune.stop()

        
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
        