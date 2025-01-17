from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Input as Inp
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adamax, Adadelta

import tensorflow as tf

from src.base.experiment.tasks.task import CELEB_A_TASK, CIFAR_10_TASK, FASHION_MNIST_TASK, MNIST_TASK
from src.base.experiment.tasks.task import ICAO_REQ

from src.m_utils.constants import SEED
from src.m_utils.stl_approach import STLApproach
from src.m_utils.mtl_approach import MTLApproach
from src.m_utils.nas_mtl_approach import NAS_MTLApproach

from src.base.experiment.training.base_models import BaseModel
from src.base.experiment.training.optimizers import Optimizer
from src.base.experiment.training.custom_base_model import CustomBaseModel

from src.base.experiment.dataset.dataset import Dataset
from src.nas.v3.mlp_search_space import MLPSearchSpaceIndicator

class ModelCreator:
    def __init__(self, config_interp):
        self.config_interp = config_interp


    def __get_optimizer(self):
        opt = None
        if self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.ADAM.name:
            opt = Adam(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'], decay=self.config_interp.mlp_params['mlp_learning_rate'] / self.config_interp.mlp_params['mlp_n_epochs'])
        elif self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.ADAM_CUST.name:
            opt = Adam(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'])
        elif self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.SGD.name:
            opt = SGD(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'])
        elif self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.SGD_NESTEROV.name:
            opt = SGD(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'], nesterov=True)
        elif self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.ADAGRAD.name:
            opt = Adagrad(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'])
        elif self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.ADAMAX.name:
            opt = Adamax(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'])
        elif self.config_interp.mlp_params['mlp_optimizer'].name == Optimizer.ADADELTA.name:
            opt = Adadelta(learning_rate=self.config_interp.mlp_params['mlp_learning_rate'])
        return opt


    def __create_base_model(self):
        baseModel = None
        W,H = self.config_interp.base_model.value['target_size']
        if self.config_interp.base_model.name == BaseModel.MOBILENET_V2.name:
            baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        elif self.config_interp.base_model.name == BaseModel.VGG19.name:
            baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        elif self.config_interp.base_model.name == BaseModel.VGG16.name:
            baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        elif self.config_interp.base_model.name == BaseModel.RESNET50_V2.name:
            baseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        elif self.config_interp.base_model.name == BaseModel.INCEPTION_V3.name:
            baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(W,H,3)), input_shape=(W,H,3))
        elif self.config_interp.base_model.name == BaseModel.CUSTOM.name:
            baseModel = CustomBaseModel(input_tensor=Inp(shape=(W,H,3)), input_shape=(W,H,3))
        return baseModel
    
        
    def create_stl_model(self, train_gen):
        baseModel = self.__create_base_model()
        
        headModel = None
        if self.config_interp.base_model.name != BaseModel.INCEPTION_V3.name:
            headModel = baseModel.output
        elif self.config_interp.base_model.name == BaseModel.INCEPTION_V3.name:
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(8, 8))(headModel)

        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        N_CLASSES = len(train_gen.class_indices.values())
        
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu", kernel_initializer=initializer)(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(N_CLASSES, activation="softmax", kernel_initializer=initializer)(headModel)
        
        model = Model(inputs=baseModel.input, outputs=headModel)
        
        opt = self.__get_optimizer()

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        return baseModel, model
    
    
    def __compile_mtl_model(self, input_layer, output_layers):
        model = Model(inputs=input_layer, outputs=output_layers)
        
        n_tasks = len(self.config_interp.tasks)

        opt = self.__get_optimizer()
        loss_list = ['sparse_categorical_crossentropy' for _ in range(n_tasks)]
        metrics_list = ['accuracy']
        loss_weights = [.1 for _ in range(n_tasks)]
 
        model.compile(loss=loss_list, loss_weights=loss_weights, optimizer=opt, metrics=metrics_list)

        return model
    
    
    def __create_branch_1(self, prev_layer, req_name, n_out, initializer):
        y = Dense(64, activation='relu', kernel_initializer=initializer)(prev_layer)
        y = Dropout(self.config_interp.mlp_params['mlp_dropout'])(y)
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
    

    def __create_branches_list_mtl_model(self, initializer, x):
        return [self.__create_branch_1(x, f'{t.value}', 2, initializer) for t in self.config_interp.tasks]

    
    def __create_mtl_model_1(self):
        baseModel = self.__create_base_model()
        
        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.config_interp.mlp_params['mlp_dropout'])(x)
        x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.config_interp.mlp_params['mlp_dropout'])(x)
        
        branches_list = self.__create_branches_list_mtl_model(initializer, x)
        
        model = self.__compile_mtl_model(baseModel.input, branches_list)

        return baseModel, model
   
    
    def __get_tasks_groups(self):
        tasks_groups = {'g0':[], 'g1':[], 'g2':[], 'g3':[]}
        if self.config_interp.dataset.name == Dataset.FVC_ICAO.name:
            tasks_groups['g0'] = [ICAO_REQ.BACKGROUND, ICAO_REQ.CLOSE, ICAO_REQ.INK_MARK, ICAO_REQ.PIXELATION,
                                    ICAO_REQ.WASHED_OUT, ICAO_REQ.BLURRED, ICAO_REQ.SHADOW_HEAD]
            tasks_groups['g1'] = [ICAO_REQ.MOUTH, ICAO_REQ.VEIL]
            tasks_groups['g2'] = [ICAO_REQ.RED_EYES, ICAO_REQ.FLASH_LENSES, ICAO_REQ.DARK_GLASSES, ICAO_REQ.L_AWAY, ICAO_REQ.FRAME_EYES,
                                    ICAO_REQ.HAIR_EYES, ICAO_REQ.EYES_CLOSED, ICAO_REQ.FRAMES_HEAVY]
            tasks_groups['g3'] = [ICAO_REQ.SHADOW_FACE, ICAO_REQ.SKIN_TONE, ICAO_REQ.LIGHT, ICAO_REQ.HAT, ICAO_REQ.ROTATION, ICAO_REQ.REFLECTION]
        elif self.config_interp.dataset.name == Dataset.MNIST.name:
            tasks_groups['g0'] = [MNIST_TASK.N_0]
            tasks_groups['g1'] = [MNIST_TASK.N_1, MNIST_TASK.N_7, MNIST_TASK.N_4]
            tasks_groups['g2'] = [MNIST_TASK.N_2, MNIST_TASK.N_3]
            tasks_groups['g3'] = [MNIST_TASK.N_5, MNIST_TASK.N_6, MNIST_TASK.N_8, MNIST_TASK.N_9]
        elif self.config_interp.dataset.name == Dataset.CIFAR_10.name:
            tasks_groups['g0'] = list(CIFAR_10_TASK)[0:2]
            tasks_groups['g1'] = list(CIFAR_10_TASK)[2:4]
            tasks_groups['g2'] = list(CIFAR_10_TASK)[4:7]
            tasks_groups['g3'] = list(CIFAR_10_TASK)[7:10]
        elif self.config_interp.dataset.name == Dataset.FASHION_MNIST.name:
            tasks_groups['g0'] = list(FASHION_MNIST_TASK)[0:2]
            tasks_groups['g1'] = list(FASHION_MNIST_TASK)[2:4]
            tasks_groups['g2'] = list(FASHION_MNIST_TASK)[4:7]
            tasks_groups['g3'] = list(FASHION_MNIST_TASK)[7:10]
        elif self.config_interp.dataset.name == Dataset.CELEB_A.name:
            tasks_groups['g0'] = list(CELEB_A_TASK)[0:10]
            tasks_groups['g1'] = list(CELEB_A_TASK)[10:20]
            tasks_groups['g2'] = list(CELEB_A_TASK)[20:30]
            tasks_groups['g3'] = list(CELEB_A_TASK)[30:42]
        else:
            raise NotImplemented()
        
        return tasks_groups


    def __create_mtl_model_2(self):
        baseModel = self.__create_base_model()
        
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)

        task_groups = self.__get_tasks_groups()
        
        br_list_0 = [self.__create_fcs_block(x, 1, req.value) for req in task_groups['g0']]
        br_list_1 = [self.__create_fcs_block(x, 1, req.value) for req in task_groups['g1']]
        br_list_2 = [self.__create_fcs_block(x, 1, req.value) for req in task_groups['g2']]
        br_list_3 = [self.__create_fcs_block(x, 1, req.value) for req in task_groups['g3']]
        
        out_branches_list = br_list_0 + br_list_1 + br_list_2 + br_list_3
        
        model = self.__compile_mtl_model(baseModel.input, out_branches_list)

        return baseModel, model
    
    
    def __create_mtl_model_3(self):
        baseModel = self.__create_base_model()
        
        x = baseModel.output
        x = self.__vgg_block(x, 1, 256, 'shared_vgg_block')
        x = Flatten()(x)
        
        split = Lambda( lambda k: tf.split(k, num_or_size_splits=4, axis=1), output_shape=(None,...))(x)
        spl_0 = tf.reshape(tensor=split[0], shape=[tf.shape(split[0])[0],1,32,32])
        spl_1 = tf.reshape(tensor=split[1], shape=[tf.shape(split[1])[0],1,32,32])
        spl_2 = tf.reshape(tensor=split[2], shape=[tf.shape(split[2])[0],1,32,32])
        spl_3 = tf.reshape(tensor=split[3], shape=[tf.shape(split[3])[0],1,32,32])
        
        tasks_groups = self.__get_tasks_groups()

        g0 = self.__vgg_block(spl_0, 2, 32, 'g0')
        g1 = self.__vgg_block(spl_1, 3, 32, 'g1')
        g2 = self.__vgg_block(spl_2, 3, 32, 'g2')
        g3 = self.__vgg_block(spl_3, 2, 32, 'g3')

        br_list_0 = [self.__create_fcs_block_2(g0, 3, req.value) for req in tasks_groups['g0']]       
        br_list_1 = [self.__create_fcs_block_2(g1, 2, req.value) for req in tasks_groups['g1']]
        br_list_2 = [self.__create_fcs_block_2(g2, 3, req.value) for req in tasks_groups['g2']]
        br_list_3 = [self.__create_fcs_block_2(g3, 3, req.value) for req in tasks_groups['g3']]
        
        out_branches_list = br_list_0 + br_list_1 + br_list_2 + br_list_3
        
        model = self.__compile_mtl_model(baseModel.input, out_branches_list)

        return baseModel, model


    def __create_nas_mtl_model_1(self, config):
        baseModel = self.__create_base_model()
        
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        
        tasks_groups = self.__get_tasks_groups()
        
        br_list_0 = [self.__create_fcs_block(x, config['n_denses_0'], t.value) for t in tasks_groups['g0']]
        br_list_1 = [self.__create_fcs_block(x, config['n_denses_1'], t.value) for t in tasks_groups['g1']]
        br_list_2 = [self.__create_fcs_block(x, config['n_denses_2'], t.value) for t in tasks_groups['g2']]
        br_list_3 = [self.__create_fcs_block(x, config['n_denses_3'], t.value) for t in tasks_groups['g3']]
        
        out_branches_list = br_list_0 + br_list_1 + br_list_2 + br_list_3
        
        model = self.__compile_mtl_model(baseModel.input, out_branches_list)

        return baseModel, model


    def __create_nas_mtl_model_2(self, config):
        baseModel = self.__create_base_model()

        x = baseModel.output
        x = self.__vgg_block(x, 1, 256, 'shared_branch')
        x = Flatten()(x)
        
        split = Lambda( lambda k: tf.split(k, num_or_size_splits=4, axis=1), output_shape=(None,...))(x)
        spl_0 = tf.reshape(tensor=split[0], shape=[tf.shape(split[0])[0],1,32,32])
        spl_1 = tf.reshape(tensor=split[1], shape=[tf.shape(split[1])[0],1,32,32])
        spl_2 = tf.reshape(tensor=split[2], shape=[tf.shape(split[2])[0],1,32,32])
        spl_3 = tf.reshape(tensor=split[3], shape=[tf.shape(split[3])[0],1,32,32])
        
        tasks_groups = self.__get_tasks_groups()

        g0 = self.__vgg_block(spl_0, config['n_convs_0'], 32, 'g0')
        g1 = self.__vgg_block(spl_1, config['n_convs_1'], 32, 'g1')
        g2 = self.__vgg_block(spl_2, config['n_convs_2'], 32, 'g2')
        g3 = self.__vgg_block(spl_3, config['n_convs_3'], 32, 'g3')

        br_list_0 = [self.__create_fcs_block_2(g0, config['n_denses_0'], req.value) for req in tasks_groups['g0']]       
        br_list_1 = [self.__create_fcs_block_2(g1, config['n_denses_1'], req.value) for req in tasks_groups['g1']]
        br_list_2 = [self.__create_fcs_block_2(g2, config['n_denses_2'], req.value) for req in tasks_groups['g2']]
        br_list_3 = [self.__create_fcs_block_2(g3, config['n_denses_3'], req.value) for req in tasks_groups['g3']]
        
        out_branches_list = br_list_0 + br_list_1 + br_list_2 + br_list_3
        
        model = self.__compile_mtl_model(baseModel.input, out_branches_list)

        return baseModel, model



    def create_model(self, train_gen=None, config=None):
        if self.config_interp.approach.value == STLApproach.STL.value:
            return self.create_stl_model(train_gen)
        elif self.config_interp.approach.value == MTLApproach.HANDCRAFTED_1.value:
            return self.__create_mtl_model_1()
        elif self.config_interp.approach.value == MTLApproach.HANDCRAFTED_2.value:
            return self.__create_mtl_model_2()
        elif self.config_interp.approach.value == MTLApproach.HANDCRAFTED_3.value:
            return self.__create_mtl_model_3()
        elif self.config_interp.approach.value == NAS_MTLApproach.APPROACH_1.value or \
                self.config_interp.approach.value == NAS_MTLApproach.APPROACH_2.value:
            return self.__create_nas_mtl_model_1(config)
        elif self.config_interp.approach.value == NAS_MTLApproach.APPROACH_3.value:
            if self.config_interp.nas_params['nas_search_space'].name == MLPSearchSpaceIndicator.SS_1.name:
                return self.__create_nas_mtl_model_1(config)
            elif self.config_interp.nas_params['nas_search_space'].name == MLPSearchSpaceIndicator.SS_2.name:            
                return self.__create_nas_mtl_model_2(config)