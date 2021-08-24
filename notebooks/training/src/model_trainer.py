import os
import shutil
import zipfile
import datetime
import numpy as np

import neptune.new as neptune

from pathlib import Path

import matplotlib.pyplot as plt

from IPython.display import display

# disable tensorflow log level infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors


from tensorflow.keras.utils import plot_model
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
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adamax, Adadelta
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import load_model

from enum import Enum

from utils.constants import SEED, ICAO_REQ

## restrict memory growth -------------------
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try:
    gpu_0 = physical_devices[0]
    tf.config.experimental.set_memory_growth(gpu_0, True) 
    #tf.config.experimental.set_virtual_device_configuration(gpu_0, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6500)])
    print(' ==> Restrict GPU memory growth: True')
except: 
    raise Exception("Invalid device or cannot modify virtual devices once initialized.")
## restrict memory growth ------------------- 


class MTLApproach(Enum):
    HAND_1 = 'handcrafted_1'
    HAND_2 = 'handcrafted_2'
    HAND_3 = 'handcrafted_3'


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
    def __init__(self, net_args, prop_args, base_model, is_mtl_model, mtl_approach, neptune_run):
        self.net_args = net_args
        self.prop_args = prop_args
        self.is_mtl_model = is_mtl_model
        self.mtl_approach = mtl_approach
        self.neptune_run = neptune_run
        self.use_neptune = True if neptune_run is not None else False
        self.base_model = base_model
        
        self.baseModel = None  # instance of base model keras/tensorflow
        
        self.is_training_model = self.prop_args['train_model']
        
        self.orig_model_experiment_id = self.prop_args['orig_model_experiment_id']
        
        self.CHECKPOINT_PATH = os.path.join('training_ckpt', 'best_model.hdf5')
        self.TRAINED_MODEL_DIR_PATH = None
        
        self.__set_model_path()
        self.__check_model_existence()
        self.__clear_checkpoints()
        self.__check_gpu_availability()
        
    
    def __set_model_path(self):
        model_path = None
        if self.orig_model_experiment_id != '':
            ds = self.prop_args['gt_names']['train_validation_test'][0].value
            aligned = 'aligned' if self.prop_args['aligned'] else 'not_aligned'
            model_type = 'single_task' if not self.is_mtl_model else 'multi_task'
            req = self.prop_args['reqs'][0].value if not self.is_mtl_model else 'multi_reqs'
            model_path = os.path.join('prev_trained_models', f'{model_type}', f'{ds}_{aligned}', f'{req}', f'{self.orig_model_experiment_id}')
        else:
            if not self.is_training_model:
                raise Exception('Insert orig_model_experiment_id in field of kwargs or train a new model!')
            else:
                model_path = os.path.join('trained_model')
        
        self.TRAINED_MODEL_DIR_PATH = model_path
    

    def __check_prev_run_fields(self):
        try:
            print('-----')
            print(' ..Checking previous experiment metadata')
            
            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            
            prev_run_req = prev_run['properties/icao_reqs'].fetch()
            prev_run_aligned = float(prev_run['properties/aligned'].fetch())
            prev_run_ds = prev_run['properties/gt_names'].fetch()

            print(f' ...Prev Exp | Req: {prev_run_req}')
            print(f' ...Prev Exp | Aligned: {prev_run_aligned}')
            print(f' ...Prev Exp | DS: {prev_run_ds}')
            
            if not self.is_mtl_model:
                cur_run_req = str([self.prop_args['reqs'][0].value])
            else:
                cur_run_req = str([req.value for req in self.prop_args['reqs']])
            cur_run_aligned = float(int(self.prop_args['aligned']))
            gt_names_formatted = {
                'train_validation': [x.value.lower() for x in self.prop_args['gt_names']['train_validation']],
                'test': [x.value.lower() for x in self.prop_args['gt_names']['test']],
                'train_validation_test': [x.value.lower() for x in self.prop_args['gt_names']['train_validation_test']]
            }
            cur_run_ds = str({'gt_names': str(gt_names_formatted)})

            print(f' ...Current Exp | Req: {cur_run_req}')
            print(f' ...Current Exp | Aligned: {cur_run_aligned}')
            print(f' ...Current Exp | DS: {cur_run_ds}')

            if prev_run_req != cur_run_req:
                raise Exception('Previous experiment Requisite field does not match current experiment Requisite field!')
            if prev_run_aligned != cur_run_aligned:
                raise Exception('Previous experiment Aligned field does not match current experiment Aligned field!')
            if prev_run_req != cur_run_req:
                raise Exception('Previous experiment DS fields does not match current experiment DS field!')

            print(' ..All checked!')
            print('-----')
        
        except Exception as e:
            raise e
        finally:
            if prev_run is not None:
                prev_run.stop()
    
    
    def __download_model(self):
        try:
            print(f'Trained model dir path: {self.TRAINED_MODEL_DIR_PATH}')
            prev_run = None
            prev_run = neptune.init(run=self.orig_model_experiment_id)

            print(f'..Downloading model from Neptune')
            print(f'..Experiment ID: {self.orig_model_experiment_id}')
            print(f'..Destination Folder: {self.TRAINED_MODEL_DIR_PATH}')

            os.mkdir(self.TRAINED_MODEL_DIR_PATH)
            destination_folder = self.TRAINED_MODEL_DIR_PATH

            prev_run['artifacts/trained_model'].download(destination_folder)
            print(' .. Download done!')

            with zipfile.ZipFile(os.path.join(destination_folder, 'trained_model.zip'), 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
            
            folder_name = [x for x in os.listdir(destination_folder) if '.zip' not in x][0]
            
            os.remove(os.path.join(destination_folder, 'trained_model.zip'))
            shutil.move(os.path.join(destination_folder, folder_name, 'variables'), destination_folder)
            shutil.move(os.path.join(destination_folder, folder_name, 'saved_model.pb'), destination_folder)
            shutil.rmtree(os.path.join(destination_folder, folder_name))

            print('.. Folders set')
            print('-----------------------------')
        except Exception as e:
            raise e
        finally:
            if prev_run is not None:
                prev_run.stop()
    
    
    def __check_model_existence(self):
        print('----')
        print('Checking model existence locally...')
        if self.is_training_model:
            print('Training a new model! Not checking model existence')
        else:
            if os.path.exists(self.TRAINED_MODEL_DIR_PATH):
                print('Model already exists locally. Not downloading!')
                print(f'Trained model dir path: {self.TRAINED_MODEL_DIR_PATH}')
                self.__check_prev_run_fields()
            else:
                self.__check_prev_run_fields()
                self.__download_model()
        print('----')
    
    
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
        
        return baseModel
    
        
    def __create_model(self, train_gen):
        self.baseModel = self.__create_base_model()
        headModel = None
        if self.base_model.name != BaseModel.INCEPTION_V3.name:
            headModel = self.baseModel.output
        elif self.base_model.name == BaseModel.INCEPTION_V3.name:
            headModel = self.baseModel.output
            headModel = AveragePooling2D(pool_size=(8, 8))(headModel)

        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        N_CLASSES = len(train_gen.class_indices.values())
        
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu", kernel_initializer=initializer)(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(N_CLASSES, activation="softmax", kernel_initializer=initializer)(headModel)
        
        self.model = Model(inputs=self.baseModel.input, outputs=headModel)
        
        opt = self.__get_optimizer()

        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    
    def __compile_mtl_model(self, input_layer, output_layers):
        self.model = Model(inputs=input_layer, outputs=output_layers)
        
        opt = self.__get_optimizer()
        n_reqs = len(self.prop_args['reqs'])
        loss_list = ['sparse_categorical_crossentropy' for x in range(n_reqs)]
        metrics_list = ['accuracy']
        loss_weights = [.1 for x in range(n_reqs)]
 
        self.model.compile(loss=loss_list, loss_weights=loss_weights, optimizer=opt, metrics=metrics_list)
    
    
    def __create_branch_1(self, prev_layer, req_name, n_out):
        y = Dense(64, activation='relu', kernel_initializer=initializer)(prev_layer)
        y = Dropout(self.net_args['dropout'])(y)
        y = Dense(n_out, activation='softmax', name=req_name, kernel_initializer=initializer)(y)
        return y
    
    def __vgg_block(self, prev_layer, num_convs, num_channels, block_name):
        x = Conv2D(num_channels, kernel_size=3, padding='same', activation='relu', name=block_name+'_0')(prev_layer)
        idx = num_convs
        while idx > 0:
            x = Conv2D(num_channels, kernel_size=3, padding='same', activation='relu', name=block_name+f'_{num_convs-idx+1}')(x)
            idx -= 1
        x = MaxPooling2D(pool_size=2, strides=2, padding="same", name=block_name+f'_{num_convs-idx+2}')(x)
        return x

    def __create_fcs_block(self, prev_layer, n_dense, req_name):
        y = Dense(64, activation='relu')(prev_layer)
        for _ in range(n_dense-1):
            y = Dense(64, activation='relu')(y)
        y = Dense(2, activation='softmax', name=req_name)(y)
        return y

    def __create_fcs_block_2(self, prev_layer, n_dense, req_name):
        y = Flatten()(prev_layer)
        y = Dense(64, activation='relu')(y)
        for _ in range(n_dense-1):
            y = Dense(64, activation='relu')(y)
        y = Dense(2, activation='softmax', name=req_name)(y)
        return y
    
    
    def __create_mtl_model(self):
        self.baseModel = self.__create_base_model()
        
        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        x = self.baseModel.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        
        branches_list = [self.__create_branch_1(x, req.value, 2) for req in self.prop_args['reqs']]
        
        self.__compile_mtl_model(self.baseModel.input, branches_list)
    
    
    def __create_mtl_model_2(self):
        self.baseModel = self.__create_base_model()
        
        x = self.baseModel.output
        x = GlobalAveragePooling2D()(x)
        
        reqs_g0 = [ICAO_REQ.BACKGROUND, ICAO_REQ.CLOSE, ICAO_REQ.INK_MARK, ICAO_REQ.PIXELATION,
                   ICAO_REQ.WASHED_OUT, ICAO_REQ.BLURRED, ICAO_REQ.SHADOW_HEAD]
        br_list_0 = [self.__create_fcs_block(x, 1, req.value) for req in reqs_g0]
        
        reqs_g1 = [ICAO_REQ.MOUTH, ICAO_REQ.VEIL]
        br_list_1 = [self.__create_fcs_block(x, 1, req.value) for req in reqs_g1]
        
        reqs_g2 = [ICAO_REQ.RED_EYES, ICAO_REQ.FLASH_LENSES, ICAO_REQ.DARK_GLASSES, ICAO_REQ.L_AWAY, ICAO_REQ.FRAME_EYES,
                   ICAO_REQ.HAIR_EYES, ICAO_REQ.EYES_CLOSED, ICAO_REQ.FRAMES_HEAVY]
        br_list_2 = [self.__create_fcs_block(x, 1, req.value) for req in reqs_g2]
        
        reqs_g3 = [ICAO_REQ.SHADOW_FACE, ICAO_REQ.SKIN_TONE, ICAO_REQ.LIGHT, 
                   ICAO_REQ.HAT, ICAO_REQ.ROTATION, ICAO_REQ.REFLECTION]
        br_list_3 = [self.__create_fcs_block(x, 1, req.value) for req in reqs_g3]
        
        out_branches_list = br_list_0 + br_list_1 + br_list_2 + br_list_3
        
        self.__compile_mtl_model(self.baseModel.input, out_branches_list)
    
    
    def __create_mtl_model_3(self):
        self.baseModel = self.__create_base_model()
        
        x = self.baseModel.output
        x = self.__vgg_block(x, 1, 256, 'shared_vgg_block')
        x = Flatten()(x)
        
        split = Lambda( lambda k: tf.split(k, num_or_size_splits=4, axis=1), output_shape=(None,...))(x)
        spl_0 = tf.reshape(tensor=split[0], shape=[tf.shape(split[0])[0],1,32,32])
        spl_1 = tf.reshape(tensor=split[1], shape=[tf.shape(split[1])[0],1,32,32])
        spl_2 = tf.reshape(tensor=split[2], shape=[tf.shape(split[2])[0],1,32,32])
        spl_3 = tf.reshape(tensor=split[3], shape=[tf.shape(split[3])[0],1,32,32])
        
        g0 = self.__vgg_block(spl_0, 2, 32, 'g0')
        reqs_g0 = [ICAO_REQ.BACKGROUND, ICAO_REQ.CLOSE, ICAO_REQ.INK_MARK, ICAO_REQ.PIXELATION,
                   ICAO_REQ.WASHED_OUT, ICAO_REQ.BLURRED, ICAO_REQ.SHADOW_HEAD]
        br_list_0 = [self.__create_fcs_block_2(g0, 3, req.value) for req in reqs_g0]
        
        g1 = self.__vgg_block(spl_1, 3, 32, 'g1')
        reqs_g1 = [ICAO_REQ.MOUTH, ICAO_REQ.VEIL]
        br_list_1 = [self.__create_fcs_block_2(g1, 2, req.value) for req in reqs_g1]
        
        g2 = self.__vgg_block(spl_2, 3, 32, 'g2')
        reqs_g2 = [ICAO_REQ.RED_EYES, ICAO_REQ.FLASH_LENSES, ICAO_REQ.DARK_GLASSES, ICAO_REQ.L_AWAY, ICAO_REQ.FRAME_EYES,
                   ICAO_REQ.HAIR_EYES, ICAO_REQ.EYES_CLOSED, ICAO_REQ.FRAMES_HEAVY]
        br_list_2 = [self.__create_fcs_block_2(g2, 3, req.value) for req in reqs_g2]
        
        g3 = self.__vgg_block(spl_3, 2, 32, 'g3')
        reqs_g3 = [ICAO_REQ.SHADOW_FACE, ICAO_REQ.SKIN_TONE, ICAO_REQ.LIGHT, 
                   ICAO_REQ.HAT, ICAO_REQ.ROTATION, ICAO_REQ.REFLECTION]
        br_list_3 = [self.__create_fcs_block_2(g3, 3, req.value) for req in reqs_g3]
        
        out_branches_list = br_list_0 + br_list_1 + br_list_2 + br_list_3
        
        self.__compile_mtl_model(self.baseModel.input, out_branches_list)
    
    
    def create_model(self, train_gen):
        print('Creating model...')
        if not self.is_mtl_model:
            self.__create_model(train_gen)
        else:
            if self.mtl_approach.value == MTLApproach.HAND_1.value:
                self.__create_mtl_model()
            elif self.mtl_approach.value == MTLApproach.HAND_2.value:
                self.__create_mtl_model_2()
            elif self.mtl_approach.value == MTLApproach.HAND_3.value:
                self.__create_mtl_model_3()

        if self.use_neptune:
            self.model.summary(print_fn=lambda x: self.neptune_run['summary/train/model_summary'].log(x))
        
        print('Model created')

        
    def model_summary(self, fine_tuned=False, print_fn=print):
        if self.is_training_model:
            self.model.summary(print_fn=print_fn)
        
            if self.use_neptune:
                if not fine_tuned:
                    self.model.summary(print_fn=lambda x: self.neptune_run['summary/train/model_summary'].log(x))
                else:
                    self.model.summary(print_fn=lambda x: self.neptune_run['summary/train/fine_tune_model_summary'].log(x))
        else:
            print('Not training a model!')
        
    
    def __get_optimizer(self):
        opt = None
        if self.net_args['optimizer'].name == Optimizer.ADAM.name:
            opt = Adam(learning_rate=self.net_args['learning_rate'], decay=self.net_args['learning_rate'] / self.net_args['n_epochs'])
        elif self.net_args['optimizer'].name == Optimizer.ADAM_CUST.name:
            opt = Adam(learning_rate=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.SGD.name:
            opt = SGD(learning_rate=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.SGD_NESTEROV.name:
            opt = SGD(learning_rate=self.net_args['learning_rate'], nesterov=True)
        elif self.net_args['optimizer'].name == Optimizer.ADAGRAD.name:
            opt = Adagrad(learning_rate=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.ADAMAX.name:
            opt = Adamax(learning_rate=self.net_args['learning_rate'])
        elif self.net_args['optimizer'].name == Optimizer.ADADELTA.name:
            opt = Adadelta(learning_rate=self.net_args['learning_rate'])
        return opt
    
    
    def __log_data(self, logs):
        if not self.is_mtl_model:
            self.neptune_run['epoch/accuracy'].log(logs['accuracy'])
            self.neptune_run['epoch/val_accuracy'].log(logs['val_accuracy'])
            self.neptune_run['epoch/loss'].log(logs['loss'])    
            self.neptune_run['epoch/val_loss'].log(logs['val_loss'])
        else:
            train_acc_list = []
            val_acc_list = []
            for req in self.prop_args['reqs']:
                self.neptune_run[f'epoch/{req.value}/accuracy'].log(logs[f'{req.value}_accuracy'])
                self.neptune_run[f'epoch/{req.value}/val_accuracy'].log(logs[f'val_{req.value}_accuracy'])
                self.neptune_run[f'epoch/{req.value}/loss'].log(logs[f'{req.value}_loss'])
                self.neptune_run[f'epoch/{req.value}/val_loss'].log(logs[f'val_{req.value}_loss'])
                self.neptune_run[f'epoch/total_loss'].log(logs['loss'])
                
                train_acc_list.append(logs[f'{req.value}_accuracy'])
                val_acc_list.append(logs[f'val_{req.value}_accuracy'])
            
            total_acc, total_val_acc = np.mean(train_acc_list), np.mean(val_acc_list)
            self.neptune_run['epoch/total_accuracy'].log(total_acc)
            self.neptune_run['epoch/total_val_accuracy'].log(total_val_acc)
            

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
            self.neptune_run['learning_rate'].log(new_lr)
            
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
        return ModelCheckpoint(self.CHECKPOINT_PATH, monitor='val_loss', save_best_only=True, mode='min',
                              verbose=1)
    
    
    def vizualize_model(self, outfile_path=None):
        display(plot_model(self.model, show_shapes=True, to_file=outfile_path))
        if self.use_neptune:
            self.neptune_run['viz/model_architecture'].upload(outfile_path)

            
    def __setup_fine_tuning(self, fine_tuned):
        if fine_tuned:
            print(' .. Fine tuning base model...')
            non_traininable_layers = self.baseModel.layers[:-2]
            
            print(f' .. Base model non trainable layers: {[l.name for l in non_traininable_layers]}')
            for layer in self.model.layers:
                if layer in non_traininable_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True
        else:
            print(' .. Not fine tuning base model...')
            base_model_layers = [l.name for l in self.baseModel.layers]
            for m_l in self.model.layers:
                 if m_l.name in base_model_layers:
                    m_l.trainable = False
        
        def p_func(line):
            if 'params' in line.lower():
                print(f'  .. {line}')
        
        self.model_summary(fine_tuned, print_fn=p_func)
            
    
    def train_model(self, train_gen, validation_gen, fine_tuned, n_epochs):
        if self.is_training_model:
            print(f'Training {self.base_model.name} network')
            
            self.__setup_fine_tuning(fine_tuned)       

            callbacks_list = []

            if self.use_neptune:
                callbacks_list.append(self.__get_log_data_callback())

            if self.net_args['optimizer'].name in [Optimizer.ADAMAX_CUST.name, Optimizer.ADAGRAD_CUST.name,
                                                   Optimizer.ADAM_CUST.name, Optimizer.SGD_CUST.name]:
                callbacks_list.append(self.__get_lr_scheduler_callback())

            if self.net_args['early_stopping'] is not None:
                callbacks_list.append(self.__get_early_stopping_callback())

            callbacks_list.append(self.__get_model_checkpoint_callback())
            
            if n_epochs is None:
                epchs = self.net_args['n_epochs']
            else:
                epchs = n_epochs
                if self.use_neptune:
                    self.neptune_run['parameters/n_epochs_fine_tuning'] = epchs

            self.H = self.model.fit(
                    train_gen,
                    steps_per_epoch=train_gen.n // self.net_args['batch_size'],
                    validation_data=validation_gen,
                    validation_steps=validation_gen.n // self.net_args['batch_size'],
                    epochs=epchs,
                    callbacks=callbacks_list)
        
        elif not self.is_training_model and self.use_neptune:
            print(f'Not training a model. Downloading data from Neptune')
            self.__get_acc_and_loss_data()
        
        else:
            print(f'Not training a model and not using Neptune!')

    
    # download accuracy and loss series data from neptune
    # (previous experiment) and log them to current experiment
    def __get_acc_and_loss_data(self):
        try:
            print(f' ..Experiment ID: {self.orig_model_experiment_id}')
            print(f' ..Downloading data from previous experiment')
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            
            if not self.is_mtl_model:
                acc_series = prev_run['epoch/accuracy'].fetch_values()['value']
                val_acc_series = prev_run['epoch/val_accuracy'].fetch_values()['value']
                loss_series = prev_run['epoch/loss'].fetch_values()['value']
                val_loss_series = prev_run['epoch/val_loss'].fetch_values()['value']
                print(f' ..Download finished')

                print(f' ..Upload data to current experiment')
                for (acc,val_acc,loss,loss_val) in zip(acc_series,val_acc_series,loss_series,val_loss_series):
                    self.neptune_run['epoch/accuracy'].log(acc)
                    self.neptune_run['epoch/val_accuracy'].log(val_acc)
                    self.neptune_run['epoch/loss'].log(loss)    
                    self.neptune_run['epoch/val_loss'].log(loss_val)
            else:
                for req in self.prop_args['reqs']:
                    req = req.value
                    acc_series = prev_run[f'epoch/{req}/accuracy'].fetch_values()['value']
                    val_acc_series = prev_run[f'epoch/{req}/val_accuracy'].fetch_values()['value']
                    loss_series = prev_run[f'epoch/{req}/loss'].fetch_values()['value']
                    val_loss_series = prev_run[f'epoch/{req}/val_loss'].fetch_values()['value']
                    print(f' ..Download finished')

                    print(f' ..Upload data to current experiment')
                    for (acc,val_acc,loss,loss_val) in zip(acc_series,val_acc_series,loss_series,val_loss_series):
                        self.neptune_run[f'epoch/{req}/accuracy'].log(acc)
                        self.neptune_run[f'epoch/{req}/val_accuracy'].log(val_acc)
                        self.neptune_run[f'epoch/{req}/loss'].log(loss)    
                        self.neptune_run[f'epoch/{req}/val_loss'].log(loss_val)

            print(f' ..Upload finished')
        except Exception as e:
            print('Error in __get_acc_and_loss_data()')
            raise e
        finally:
            prev_run.stop()
        
    
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
                self.neptune_run['viz/train/training_curves'].upload(f)
        
        elif not self.is_training_model and self.use_neptune:
            print('Not training a model. Downloading plot from Neptune')
            self.__get_training_curves()
        
        else:
            print('Not training a model and not using Neptune!')
    
    # download training curves plot from Neptune previous experiment
    # and upload it to the current one
    def __get_training_curves(self):
        try:
            print(f' ..Experiment ID: {self.orig_model_experiment_id}')
            print(f' ..Downloading plot from previous experiment')
            prev_run = neptune.init(run=self.orig_model_experiment_id)
            prev_run['viz/train/training_curves'].download()
            print(f' ..Download finished')

            print(f' ..Uploading plot')
            self.neptune_run['viz/train/training_curves'].upload('training_curves.png')
            print(f' ..Upload finished')
        except Exception as e:
            print('Error in __get_training_curves()')
            raise e
        finally:
            prev_run.stop()
    

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
            def zipdir(path):
                outfile_path = 'trained_model.zip'
                with zipfile.ZipFile(outfile_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            filename = os.path.join(root, file)
                            arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                            if self.orig_model_experiment_id != "" and self.orig_model_experiment_id in arcname:
                                arcname = arcname.replace(self.orig_model_experiment_id, 'trained_model')
                            ziph.write(filename, arcname)
                return outfile_path
            
            print('Saving model')

            self.model.save(self.TRAINED_MODEL_DIR_PATH)
            print('..Model saved')
            print(f'...Model path: {self.TRAINED_MODEL_DIR_PATH}')

            if self.use_neptune:
                print('Saving model to neptune')
                trained_model_zip_path = zipdir(self.TRAINED_MODEL_DIR_PATH)
                print(f' ..Uploading file {trained_model_zip_path}')
                self.neptune_run['artifacts/trained_model'].upload(trained_model_zip_path)
                print('Model saved into Neptune')

            self.model.training = False

            print('Saving process finished')
        else:
            print('Not saving trained model!')
        