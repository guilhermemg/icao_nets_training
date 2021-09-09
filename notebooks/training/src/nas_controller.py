
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
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adamax, Adadelta
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D


class NASController:
    def __init__(self, prop_args, nas_params, base_model, is_mtl_model, neptune_run):
        self.prop_args = prop_args
        self.nas_params = nas_params
        self.base_model = base_model
        self.is_mtl_model = is_mtl_model
        self.neptune_run = neptune_run
        
        self.model = None
            
    
    # # receives current topology and adds basic block
    # # returns new topology
    # def __add_basic_block(self, block_num, step):
    #     print(f'Block Num: {block_num}')
    #     print(f'Step: {step}')
        
    #     n_reqs = len(self.prop_args['reqs'])
        
    #     cur_topology = self.model.layers[-2]

    #     if block_num == 0 and step == 0:
    #         cur_topology = GlobalAveragePooling2D()(cur_topology.output)
        
    #     new_topology = Conv2D(32, (3,3), activation='relu')(cur_topology)
    #     new_topology = Flatten()(new_topology)
    #     new_topology = Dense(n_reqs, activation='softmax', name=f'new_out_{block_num}_{step}')(new_topology)
        
    #     self.model = Model(inputs=self.model.input, outputs=new_topology)
        
    #     opt = Adamax(lr=1e-3)
    #     loss_list = ['sparse_categorical_crossentropy' for x in range(n_reqs)]
    #     metrics_list = ['accuracy']
    #     loss_weights = [.1 for x in range(n_reqs)]
 
    #     self.model.compile(loss=loss_list, loss_weights=loss_weights, optimizer=opt, metrics=metrics_list)
    
    #     self.model.summary()
        
    #     print('=====================================================')
    #     print('=====================================================')
        
    
    # def __train_topology(self, train_gen, validation_gen):
    #     self.H = self.model.fit(
    #                 train_gen,
    #                 steps_per_epoch=train_gen.n // 64,
    #                 validation_data=validation_gen,
    #                 validation_steps=validation_gen.n // 64,
    #                 epochs=2)
    
    
    # def run_nas(self, train_gen, validation_gen):
    #     block_num = 0
    #     while block_num < self.nas_params['max_blocks_per_branch']:
    #         for step in range(self.nas_params['max_train_steps_per_op']):
    #             self.__add_basic_block(block_num, step)
    #             self.__train_topology(train_gen, validation_gen)
    #         block_num += 1
        
#         initializer = RandomNormal(mean=0., stddev=1e-4, seed=SEED)
        
#         x = baseModel.output
#         x = GlobalAveragePooling2D()(x)
#         #x = Flatten()(x)
        
#         x = Dense(256, activation='relu', kernel_initializer=initializer)(x)
#         x = Dropout(self.net_args['dropout'])(x)
#         x = Dense(128, activation='relu', kernel_initializer=initializer)(x)
#         x = Dropout(self.net_args['dropout'])(x)
        
#         branches_list = [__create_branch(x, req.value, 2) for req in self.prop_args['reqs']]
        
#         self.model = Model(inputs=baseModel.input, outputs=branches_list)
        
#         opt = Adamax(lr=1e-4)
#         n_reqs = len(self.prop_args['reqs'])
#         loss_list = ['sparse_categorical_crossentropy' for x in range(n_reqs)]
#         metrics_list = ['accuracy']
#         loss_weights = [.1 for x in range(n_reqs)]
 
        #self.model.compile(loss=loss_list, loss_weights=loss_weights, optimizer=opt, metrics=metrics_list)
                
        #net_mng = NASNetManager()


    def select_topology(self):
        config = [1,1,1,1]   # TODO 
        return config
    

    def evaluate_topology(self, history_logs):
        pass