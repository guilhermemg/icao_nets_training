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
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adamax, Adadelta

import tensorflow as tf

from enum import Enum

from utils.constants import SEED, ICAO_REQ
from base_models import BaseModel
from optimizers import Optimizer


class MTLApproach(Enum):
    HAND_1 = 'handcrafted_1'
    HAND_2 = 'handcrafted_2'
    HAND_3 = 'handcrafted_3'


class ModelCreator:
    def __init__(self, net_args, prop_args, base_model, mtl_approach, is_mtl_model):
        self.net_args = net_args
        self.prop_args = prop_args
        self.mtl_approach = mtl_approach
        self.is_mtl_model = is_mtl_model
        self.base_model = base_model   # enum tpe BaseModel
    

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
    
        
    def create_stl_model(self, train_gen):
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
        
        model = Model(inputs=baseModel.input, outputs=headModel)
        
        opt = self.__get_optimizer()

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        return baseModel, model
    
    
    def __compile_mtl_model(self, input_layer, output_layers):
        model = Model(inputs=input_layer, outputs=output_layers)
        
        opt = self.__get_optimizer()
        n_reqs = len(self.prop_args['reqs'])
        loss_list = ['sparse_categorical_crossentropy' for x in range(n_reqs)]
        metrics_list = ['accuracy']
        loss_weights = [.1 for x in range(n_reqs)]
 
        model.compile(loss=loss_list, loss_weights=loss_weights, optimizer=opt, metrics=metrics_list)

        return model
    
    
    def __create_branch_1(self, prev_layer, req_name, n_out, initializer):
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
    
    
    def create_mtl_model(self):
        baseModel = self.__create_base_model()
        
        initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
        x = baseModel.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
        x = Dropout(self.net_args['dropout'])(x)
        
        branches_list = [self.__create_branch_1(x, req.value, 2, initializer) for req in self.prop_args['reqs']]
        
        model = self.__compile_mtl_model(baseModel.input, branches_list)

        return baseModel, model
    
    
    def create_mtl_model_2(self):
        baseModel = self.__create_base_model()
        
        x = baseModel.output
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
        
        model = self.__compile_mtl_model(baseModel.input, out_branches_list)

        return baseModel, model
    
    
    def create_mtl_model_3(self):
        baseModel = self.__create_base_model()
        
        x = baseModel.output
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
        
        model = self.__compile_mtl_model(baseModel.input, out_branches_list)

        return baseModel, model


    def create_model(self, train_gen=None):
        if not self.is_mtl_model:
            return self.create_stl_model(train_gen)
        else:
            if self.mtl_approach.value == MTLApproach.HAND_1.value:
                return self.create_mtl_model()
            elif self.mtl_approach.value == MTLApproach.HAND_2.value:
                return self.create_mtl_model_2()
            elif self.mtl_approach.value == MTLApproach.HAND_3.value:
                return self.create_mtl_model_3()