import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.initializers import Zeros

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

class CNN_Keras:
    def __init__(self, num_classes, cnn_config, cnn_drop_rate, batch_size):
        self.num_classes = num_classes
        self.cnn_config = cnn_config
        self.cnn_drop_rate = cnn_drop_rate
        self.batch_size = batch_size
        
        print(f'num_classes: {self.num_classes}')
        print(f'cnn_config: {self.cnn_config}')
        print(f'cnn_drop_rate: {self.cnn_drop_rate}')
        print(f'batch_size: {self.batch_size}')
        
        
    def build_model(self):
        cnn = [c[0] for c in self.cnn_config]
        cnn_num_filters = [c[1] for c in self.cnn_config]
        max_pool_ksize = [c[2] for c in self.cnn_config]
        
        print(f'cnn: {cnn}')
        print(f'cnn_num_filters: {cnn_num_filters}')
        print(f'max_pool_ksize: {max_pool_ksize}')
        
        self.model = Sequential()
        
        for idd, filter_size in enumerate(cnn):
            conv_out = None
            if idd == 0:
                conv_out = Conv2D(
                    filters=64,
                    kernel_size=(3,3),
                    name="conv_out_"+str(idd),
                    activation='relu',
                    input_shape=(28,28,1)
                )
            else:
                conv_out = Conv2D(
                    filters=64,
                    kernel_size=(3,3),
                    name="conv_out_"+str(idd),
                    activation='relu'
                )
            self.model.add(conv_out)
            
            pool_out = MaxPooling2D(
                pool_size=(2,2),
                name="max_pool_"+str(idd)
            )
            self.model.add(pool_out)
            
            drop_out = Dropout(0.85)
            self.model.add(drop_out)
        
        flatten_pred_out = Flatten()
        self.model.add(flatten_pred_out)
        
        softmax = Dense(self.num_classes, activation='softmax')
        self.model.add(softmax)
        
        print(self.model.summary())
        
    
    def compile_model(self):
        opt = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    