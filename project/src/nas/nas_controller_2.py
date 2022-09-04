
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, RandomUniform, GlorotNormal

from src.nas.gen_nas_controller import GenNASController
from src.m_utils.constants import SEED

class NASController_2(GenNASController):
    def __init__(self, model_trainer, model_evaluator, config_interp, neptune_utils):
        super().__init__(model_trainer, model_evaluator, config_interp, neptune_utils)       
        
        self.baseline = None
        self.lstm_cell_units = 32
        self.baseline_decay = 0.999
        self.opt = Adam(lr=0.00035, decay=1e-3, amsgrad=True)
        self.controller_batch_size = self.config_interp.nas_params['controller_batch_size']
        self.controller_epochs = self.config_interp.nas_params['controller_epochs']

        rng = np.random.default_rng(SEED)
        self.input_x = np.array([[rng.random(4)]])

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
        print('   Compiling Controller RNN...')

        def __controller_loss(y_true, y_pred):
            import tensorflow as tf
            
            if self.baseline is None:
                #print('baseline is None')
                self.baseline = 0
            else:
                #print('baseline is not None')
                self.baseline -= (1 - self.baseline_decay) * (self.baseline - self.reward)
            
            #print(f' ..baseline: {self.baseline} | baseline_decay: {self.baseline_decay} | reward: {self.reward}')
            #tf.print(f' ..y_pred: ', y_pred)
            
            l = y_pred * (self.reward - self.baseline)
            #tf.print(f' ..loss: ', l)
            
            return l

        self.controller_rnn.compile(loss=__controller_loss, optimizer=self.opt)

        print('    ..done!')


    def __train_controller_rnn(self, targets):
        print(f' .. training controller rnn ..')
        print(f'  .. targets: {targets}')
        self.__compile_controller_rnn()
        H = self.controller_rnn.fit(
            self.input_x,
            targets,
            epochs=self.controller_epochs,
            batch_size=self.controller_batch_size,
            verbose=0)
        
        self.__visualize_history(H)
        

    def __visualize_history(self, history):
        from matplotlib import pyplot as plt

        f,ax = plt.subplots(1,2, figsize=(10,5))
        f.suptitle(f'-----Trial Training Result-----')

        ax[1].plot(history.history['loss'])
        #ax[1].plot(history.history['val_loss'])
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('loss')
        ax[1].set_xlabel('epoch')
        ax[1].legend(['train', 'validation'])

        plt.show()


    
    def __evaluate_contoller_rnn_training(self, targets):
        print('Evaluating targets...')
        loss = round(self.controller_rnn.evaluate(self.input_x, targets), 8)
        print(f'  Loss: {loss}')


    def __softmax_predict(self, input_x):
        print('Softmax predict on new input_x')
        #self.__compile_controller_rnn()
        return self.controller_rnn.predict(input_x)


    def __convert_pred_to_ydict(self, controller_pred):
        vals = controller_pred[0]
        final_vals = []
        for v in vals:
            if v < 0.2:
                final_vals.append(1)
            elif v >= 0.2 and v < 0.4:
                final_vals.append(2)
            elif v >= 0.4 and v < 0.6:
                final_vals.append(3)
            elif v >= 0.6 and v < 0.8:
                final_vals.append(4)
            elif v >= 0.8 and v < 1.0:
                final_vals.append(5)

        config = {f'n_denses_0': final_vals[0], f'n_denses_1': final_vals[1], f'n_denses_2': final_vals[2], f'n_denses_3': final_vals[3]}
        return config


    def select_config(self):
        print(' Selecting new config...')

        controller_pred = None        
        if self.memory.is_empty():
            print('  Memory is empty')
            self.__compile_controller_rnn()
            controller_pred = self.__softmax_predict(self.input_x)
        else:
            print('  Memory is not empty')

            last_trial = self.memory.get_last_trial()
            last_trial_conf = last_trial.get_config()
            last_trial_orig_vals = last_trial.get_orig_vals()
            print(f'  Last Trial Conf: {last_trial_conf}')
            print(f'  Last Trial Orig Vals: {last_trial_orig_vals}')
            
            # entry = np.array([[[last_trial_conf['n_denses_0'],last_trial_conf['n_denses_1'],
            #                     last_trial_conf['n_denses_2'],last_trial_conf['n_denses_3']]]])
            entry = np.array([last_trial_orig_vals])
            print(f'  LSTM entry: {entry}')
            
            controller_pred = self.__softmax_predict(entry)
        
        print(f' controller_pred: {controller_pred}')
        config = self.__convert_pred_to_ydict(controller_pred)        
        return controller_pred, config
    

    def run_nas_trial(self, trial_num, train_gen, validation_gen):
        print('\n' + '='*20 + ' STARTING NEW TRIAL ' + '='*20)

        self.cur_trial = self.create_new_trial(trial_num)
        controller_pred, config = self.select_config()
        self.cur_trial.set_config(config)
        self.cur_trial.set_orig_vals(controller_pred)
            
        final_eval = self.train_child_architecture(train_gen, validation_gen)

        self.reward = final_eval['final_ACC']

        self.set_config_eval(final_eval)

        self.__train_controller_rnn(controller_pred)
        self.__evaluate_contoller_rnn_training(controller_pred)

        self.log_trial()
        self.finish_trial()

        print('='*20 + 'FINISHING TRIAL' + '='*20 + '\n')


    def __get_weight_initializer(self, initializer=None, seed=None):
        if initializer is None:
            return HeNormal()
        elif initializer == "lstm":
            return RandomUniform(minval=-0.1, maxval=0.1)
        else:
            return GlorotNormal()