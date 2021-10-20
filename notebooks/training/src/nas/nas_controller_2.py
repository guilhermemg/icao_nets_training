
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, RandomUniform, GlorotNormal

from nas.gen_nas_controller import GenNASController
from utils.constants import SEED

class NASController_2(GenNASController):
    def __init__(self, model_trainer, model_evaluator, nas_params, neptune_run, use_neptune):
        super().__init__(model_trainer, model_evaluator, nas_params, neptune_run, use_neptune)       
        
        self.baseline = None
        self.lstm_cell_units = 32
        self.baseline_decay = 0.999
        self.opt = Adam(lr=0.00035, decay=1e-3, amsgrad=True)
        self.controller_batch_size = self.nas_params['controller_batch_size']
        self.controller_epochs = self.nas_params['controller_epochs']

        self.input_x = np.array([[[SEED,SEED,SEED,SEED]]])

        self.reward = 0

        self.controller_rnn = None
        
        self.__generate_controller_rnn()


    def __generate_controller_rnn(self):
        controller_input = Input(shape=(1,4,))        

        cell = LSTMCell(
            self.lstm_cell_units,
            kernel_initializer=self.__get_weight_initializer(initializer="lstm"),
            recurrent_initializer=self.__get_weight_initializer(initializer="lstm"))
        
        x = RNN(cell, return_state=True)(controller_input)
        y = Dense(4)(x[0])
        y = Activation(activation="softmax")(y)
        
        self.controller_rnn = Model(inputs=controller_input, outputs=y)

    
    def __compile_controller_rnn(self):
        def __controller_loss(y_true, y_pred):
            if self.baseline is None:
                self.baseline = 0
            else:
                self.baseline -= (1 - self.baseline_decay) * (self.baseline - self.reward)
            return y_pred * (self.reward - self.baseline)

        self.controller_rnn.compile(loss=__controller_loss, optimizer=self.opt)


    def __train_controller_rnn(self, targets):
        print(f' .. training controller rnn ..')
        print(f'  .. targets: {targets}')
        self.__compile_controller_rnn()
        self.controller_rnn.fit(
            self.input_x,
            targets,
            epochs=self.controller_epochs,
            batch_size=self.controller_batch_size,
            verbose=1)


    def __softmax_predict(self):
        self.__compile_controller_rnn()
        return self.controller_rnn.predict(self.input_x)


    def __convert_pred_to_ydict(self, controller_pred):
        vals = controller_pred[0]
        vals = np.interp(vals, (vals.min(), vals.max()), (1, self.MAX_BLOCKS_PER_BRANCH))
        vals = vals.astype('int')
        config = {f'n_denses_0': vals[0], f'n_denses_1': vals[1], f'n_denses_2': vals[2], f'n_denses_3': vals[3]}
        return config


    def select_config(self):
        print(' selecting new config...')
        controller_pred = self.__softmax_predict()
        print(f' controller_pred: {controller_pred}')
        config = self.__convert_pred_to_ydict(controller_pred)     
        self.cur_trial.set_config(config)
        return controller_pred, config
    

    def run_nas_trial(self, trial_num, train_gen, validation_gen):
        print('+'*20 + ' STARTING NEW TRAIN ' + '+'*20)

        self.create_new_trial(trial_num)

        controller_pred, config = self.select_config()
            
        final_eval = self.train_child_architecture(trial_num, train_gen, validation_gen, config)

        self.set_config_eval(final_eval)

        self.__train_controller_rnn(controller_pred)

        self.log_trial()
        self.finish_trial()

        print('-'*20 + 'FINISHING TRAIN' + '-'*20)


    def __get_weight_initializer(self, initializer=None, seed=None):
        if initializer is None:
            return HeNormal()
        elif initializer == "lstm":
            return RandomUniform(minval=-0.1, maxval=0.1)
        else:
            return GlorotNormal()