import os
import neptune
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt


from keras.utils.vis_utils import plot_model

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as prep_input_inceptionv3
from tensorflow.keras.applications.vgg19 import preprocess_input as prep_input_vgg19
from tensorflow.keras.applications.vgg16 import preprocess_input as prep_input_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as prep_input_resnet50v2

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
from tensorflow.keras.models import load_model

from enum import Enum

from utils.constants import SEED


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


class ModelTrainer:
    def __init__(self, net_args, prop_args, base_model, is_mtl_model, use_neptune):
        self.net_args = net_args
        self.prop_args = prop_args
        self.is_mtl_model = is_mtl_model
        self.use_neptune = use_neptune
        self.base_model = base_model
        
        self.is_training_model = self.prop_args['train_model']
        
        self.CHECKPOINT_PATH = os.path.join('training_ckpt', 'best_model.hdf5')
        
        model_name = self.prop_args['model_name']
        if model_name != '':
            self.TRAINED_MODEL_DIR_PATH = os.path.join('prev_trained_models', model_name)
        else:
            if not self.is_training_model:
                print('Error! Insert model name in field of kwargs')
            else:
                self.TRAINED_MODEL_DIR_PATH = os.path.join('trained_model')
        
        self.__clear_checkpoints()
        self.__check_gpu_availability()
        
    
    def __check_gpu_availability(self):
        print('------------------------------')
        print('Checking GPU availability')
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print(' ..GPU is available!')
        else:
            print(' ..GPU is NOT available!')
        print('------------------------------')
    
    
    def __clear_checkpoints(self):
        ckpts_dir_path = Path(self.CHECKPOINT_PATH.split(os.path.sep)[0])
        if(not os.path.exists(ckpts_dir_path)):
            ckpts_dir_path.mkdir(parents=True)
        
        if os.path.exists(self.CHECKPOINT_PATH):
            os.remove(self.CHECKPOINT_PATH)
    
    
    def __create_base_model(self):
        baseModel = None
        W,H = self.base_model.value['target_size']
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
        
        for layer in baseModel.layers:
            layer.trainable = False
        
        return baseModel
    
        
    def __create_model(self, train_gen):
        baseModel = self.__create_base_model()
        headModel = None
        if self.base_model.name != BaseModel.INCEPTION_V3.name:
            headModel = baseModel.output
        elif self.base_model.name == BaseModel.INCEPTION_V3.name:
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(8, 8))(headModel)

        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        N_CLASSES = len(train_gen.class_indices.values())
        
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu", kernel_initializer=initializer)(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(N_CLASSES, activation="softmax", kernel_initializer=initializer)(headModel)

        self.model = Model(inputs=baseModel.input, outputs=headModel)
        
        opt = self.__get_optimizer()

        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    
    def __create_mtl_model(self):
        def __create_branch(shared_branch, req_name, n_out):
            y = Dense(64, activation='relu', kernel_initializer=initializer)(shared_branch)
            y = Dropout(self.net_args['dropout'])(y)
            y = Dense(n_out, activation='softmax', name=req_name, kernel_initializer=initializer)(y)
            return y
        
        baseModel = self.__create_base_model()
        
        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)
        
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        
        branches_list = [__create_branch(x, req.value, 2) for req in self.prop_args['reqs']]
        
        self.model = Model(inputs=baseModel.input, outputs=branches_list)
        
        opt = self.__get_optimizer()
        n_reqs = len(self.prop_args['reqs'])
        loss_list = ['sparse_categorical_crossentropy' for x in range(n_reqs)]
        metrics_list = ['accuracy']
        loss_weights = [.1 for x in range(n_reqs)]
 
        self.model.compile(loss=loss_list, loss_weights=loss_weights, optimizer=opt, metrics=metrics_list)
        
        
    def create_model(self, train_gen):
        print('Creating model...')
        if not self.is_mtl_model:
            self.__create_model(train_gen)
        else:
            self.__create_mtl_model()

        if self.use_neptune:
            self.model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
        
        print('Model created')

    
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
        if not self.is_mtl_model:
            neptune.log_metric('epoch_accuracy', logs['accuracy'])
            neptune.log_metric('epoch_val_accuracy', logs['val_accuracy'])
            neptune.log_metric('epoch_loss', logs['loss'])    
            neptune.log_metric('epoch_val_loss', logs['val_loss'])
        else:
            train_acc_list = []
            val_acc_list = []
            for req in self.prop_args['reqs']:
                neptune.log_metric(f'epoch_accuracy_{req.value}', logs[f'{req.value}_accuracy'])
                neptune.log_metric(f'epoch_val_{req.value}_accuracy', logs[f'val_{req.value}_accuracy'])
                neptune.log_metric(f'epoch_loss_{req.value}', logs[f'{req.value}_loss'])
                neptune.log_metric(f'epoch_val_{req.value}_loss', logs[f'val_{req.value}_loss'])
                neptune.log_metric(f'total_loss', logs['loss'])
                
                train_acc_list.append(logs[f'{req.value}_accuracy'])
                val_acc_list.append(logs[f'val_{req.value}_accuracy'])
            
            total_acc, total_val_acc = np.mean(train_acc_list), np.mean(val_acc_list)
            neptune.log_metric('epoch_total_accuracy', total_acc)
            neptune.log_metric('epoch_total_val_accuracy', total_val_acc)
            

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
        return ModelCheckpoint(self.CHECKPOINT_PATH, monitor='val_loss', save_best_only=True, mode='min')
    
    
#     def __get_my_callback(self):
#         class MyCallback(tf.keras.callbacks.Callback): 
#             def __init__(self, val_gen):
#                 self.val_gen = val_gen
#                 self.out_file_path = 'output/out.csv'

#             def __clean_out_file(self):
#                 if os.path.exists(self.out_file_path):
#                     os.remove(self.out_file_path)

#             def on_epoch_end(self, epoch, logs={}): 
#                 if epoch == 0:
#                     self.__clean_out_file()

#         #         print(f'Validation Accuracy: {self.model.evaluate(self.val_gen, batch_size=BS)}')

#                 with open(self.out_file_path,'a') as f:
#                     predIxs = self.model.predict(self.val_gen, batch_size=BS)
#                     Y = self.val_gen.labels
#                     Y_hat = np.argmax(predIxs, axis=1)
#                     for idx,(y,y_h) in enumerate(zip(Y,Y_hat)):
#                         if epoch == 0 and idx == 0:
#                             f.writelines('epoch,idx,y,y_hat\n')
#                         f.writelines(f'{epoch+1},{idx},{y},{y_h}\n')   
        
#         return MyCallback(self.validation_gen)


    def vizualize_model(self, outfile_path=None):
        display(plot_model(self.model, show_shapes=True, to_file=outfile_path))

    
    def train_model(self, train_gen, validation_gen):
        if self.is_training_model:
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
                    train_gen,
                    steps_per_epoch=train_gen.n // self.net_args['batch_size'],
                    validation_data=validation_gen,
                    validation_steps=validation_gen.n // self.net_args['batch_size'],
                    epochs=self.net_args['n_epochs'],
                    callbacks=callbacks_list)
        else:
            print('Not training a model')

    
    def draw_training_history(self):
        if self.is_training_model:
            if not self.is_mtl_model:
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

            else:
                f,ax = plt.subplots(2,2, figsize=(20,25))
                f.suptitle(f'-----{self.base_model.name}-----')

                for idx,req in enumerate(self.prop_args['reqs']):
                    ax[0][0].plot(self.H.history[f'{req.value}_accuracy'])
                    ax[0][1].plot(self.H.history[f'val_{req.value}_accuracy'])

                    ax[1][0].plot(self.H.history[f'{req.value}_loss'])
                    ax[1][1].plot(self.H.history[f'val_{req.value}_loss'])

                ax[1][1].plot(self.H.history['loss'], color='red', linewidth=2.0) # total loss

                ax[0][0].set_title('Model Accuracy - Train')
                ax[0][1].set_title('Model Accuracy - Validation')

                ax[0][0].set_ylabel('accuracy')
                ax[0][1].set_ylabel('accuracy')
                ax[0][0].set_xlabel('epoch')
                ax[0][1].set_xlabel('epoch')

                ax[0][0].set_ylim([0,1.1])
                ax[0][1].set_ylim([0,1.1])

                ax[1][0].set_title('Model Loss - Train')
                ax[1][1].set_title('Model Loss - Validation')

                ax[1][0].set_ylabel('loss')
                ax[1][1].set_ylabel('loss')

                ax[1][0].set_xlabel('epoch')
                ax[1][1].set_xlabel('epoch')

                ax[1][0].set_ylim([0,1.5])
                ax[1][1].set_ylim([0,1.5])

                legends = [r.value for r in self.prop_args['reqs']]
                ax[0][0].legend(legends, ncol=4)
                ax[0][1].legend(legends, ncol=4)
                ax[1][0].legend(legends, ncol=4)
                ax[1][1].legend(legends, ncol=4)
            
            plt.show()

            if self.use_neptune:
                neptune.send_image('training_curves.png',f)
        else:
            print('Not training a model')
    
    
    def load_checkpoint(self, chkp_name):
        self.__create_model()
        self.model.load_weights(chkp_name)
    
    
    def load_best_model(self):
        print('..Loading best model')
        
        if self.is_training_model:
            if os.path.isfile(self.CHECKPOINT_PATH):
                self.model.load_weights(self.CHECKPOINT_PATH)
                print('..Checkpoint weights loaded')
            else:
                print('Checkpoint not found')
        else:
            self.model = load_model(self.TRAINED_MODEL_DIR_PATH)
            print('..Model loaded')
            print(f'...Model path: {self.TRAINED_MODEL_DIR_PATH}')
            
    
    def save_trained_model(self):
        if self.prop_args['save_trained_model']:
            print('Saving model')

            self.model.save(self.TRAINED_MODEL_DIR_PATH)
            print('..Model saved')
            print(f'...Model path: {self.TRAINED_MODEL_DIR_PATH}')

            if self.use_neptune:
                print('..Saving model to neptune..')
                for item in os.listdir(self.TRAINED_MODEL_DIR_PATH):
                    neptune.log_artifact(os.path.join(self.TRAINED_MODEL_DIR_PATH, item))
                print('Model saved into Neptune')

            self.model.training = False

            print('Saving process finished')
        else:
            print('Not saving trained model!')
        