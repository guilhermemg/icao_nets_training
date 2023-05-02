import os
import random
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from src.base.experiment.training.optimizers import Optimizer


class NASController_4:
    def __init__(self, config_interp):        
        self.config_interp = config_interp

        self.max_len                    = self.config_interp.mlp_params['max_architecture_length']
        self.min_task_group_size        = self.config_interp.mlp_params['min_task_group_size']
        self.controller_lstm_dim        = self.config_interp.controller_params['controller_lstm_dim']
        self.controller_optimizer       = self.config_interp.controller_params['controller_optimizer']
        self.controller_lr              = self.config_interp.controller_params['controller_learning_rate']
        self.controller_decay           = self.config_interp.controller_params['controller_decay']
        self.controller_momentum        = self.config_interp.controller_params['controller_momentum']
        self.use_predictor              = self.config_interp.controller_params['controller_use_predictor']
        self.controller_loss_alpha      = self.config_interp.controller_params['controller_loss_alpha']
        self.controller_train_epochs    = self.config_interp.controller_params['controller_training_epochs']
        self.controller_sampling_epochs = self.config_interp.controller_params['controller_sampling_epochs']

        self.controller_weights_path = 'LOGS/controller_weights.h5'

        self.n_tasks = len(self.config_interp.tasks)

        #self.controller_classes = len(self.vocab) + 1
        self.controller_classes = 4

        self.controller_batch_size = self.config_interp.controller_params['controller_batch_size']
        self.controller_input_shape = (1, self.config_interp.mlp_params['max_architecture_length'] - 1)
        self.controller_use_predictor = self.config_interp.controller_params['controller_use_predictor']
        
        self.nas_data_history = None
        self.search_space = None
        self.search_space_size = None

        self.__create_search_space()


        if not self.controller_use_predictor:
            self.controller_model = self.__create_control_model(self.controller_input_shape)
        else:
            self.controller_model = self.__create_hybrid_control_model(self.controller_input_shape, self.controller_batch_size)


    def __create_search_space(self):
        all_archs = [[i,j,p,q] for i in range(0,5) for j in range(0,5) for p in range(0,5) for q in range(0,5)]
        print(f'all_archs[:10]: {all_archs[:10]}')
        
        search_space_size = len(all_archs)
        print(f'search_space_size: {search_space_size}')

        self.search_space = all_archs
        self.search_space_size = len(self.search_space)


    def build_new_arch(self, prev_arch):
        inp = np.array(prev_arch).reshape(1,1,4)
        
        print(f'input: {inp}')
        
        prob_list = self.controller_model.predict(inp)
        
        prob_list = prob_list[0][0]
        print(f'prob_list[:5]: {prob_list[:5]}')
        
        chose_idx = np.random.choice(range(self.search_space_size), size=1, replace=False, p=prob_list)[0]
        print(f'chose_idx: {chose_idx}')
        
        new_arch = self.search_space[chose_idx]
        print(f'new_arch: {new_arch}')

        return new_arch


    def __prepare_controller_data(self, nas_data_history):
        self.nas_data_history = nas_data_history

        print('Preparing controller data...')
        print(f' ..nas_data_history: {nas_data_history}')
        
        xc = np.array([[item[0].to_numbers() for item in nas_data_history]])       
        xc = xc.reshape(self.controller_batch_size, 1, self.max_len - 1)
        print(f' ..xc: {xc}')
        print(f' ..xc.shape: {xc.shape}')

        #yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        yc = np.zeros((self.controller_batch_size, 1, self.max_len - 1))
        print(f' ..yc: {yc}')
        print(f' ..yc.shape: {yc.shape}')
        
        val_acc_target = [item[1] for item in nas_data_history]
        print(f' ..val_acc_target: {val_acc_target}')
        print(f' ..val_acc_target.length: {len(val_acc_target)}')
        
        return xc, yc, val_acc_target


    def get_discounted_reward(self, rewards):
        # initialise discounted reward array
        discounted_r = np.zeros_like(rewards, dtype=np.float32)

        # every element in the discounted reward array
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.

            # will need us to iterate over all rewards from t to T
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1
            
            # add values to the discounted reward array
            discounted_r[t] = running_add

        # normalize discounted reward array    
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()

        return discounted_r


    # loss function based on discounted reward for policy gradients
    def custom_loss(self, target, output):
        # define baseline for rewards and subtract it from all validation accuracies to get reward.
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.nas_data_history[-self.controller_batch_size:]]).reshape(
           self.controller_batch_size, 1)
        
        # get discounted reward
        discounted_reward = self.get_discounted_reward(reward)

        # multiply discounted reward by log likelihood of actions to get loss function
        loss = - K.log(output) * discounted_reward[:, None]

        return loss


    def train_model_controller(self, nas_data_history):
        print('Training controller model...')

        nas_data_history = nas_data_history[-self.controller_batch_size:]

        xc, yc, val_acc_target = self.__prepare_controller_data(nas_data_history)

        if self.use_predictor:
            self.__train_hybrid_control_model(
                                            xc,
                                            yc,
                                            val_acc_target,
                                            self.custom_loss,
                                            self.controller_batch_size,
                                            self.controller_train_epochs)
        else:
            self.__train_control_model(
                                     xc,
                                     yc,
                                     self.custom_loss,
                                     self.controller_batch_size,
                                     self.controller_train_epochs)
        
        print(' ..Controller model trained!')


    def __check_sequence_validity_BAK(self, sequence):
        decoded_seq = self.decode_sequence(sequence)
        
        n_groups = len(set([x for x in decoded_seq.keys() if 'g' in x]))
        n_denses = len([x for x in decoded_seq.keys() if 'n_denses' in x])

        print(f' .Decoded seq: {decoded_seq}')
        if n_groups != n_denses:
            print(f' ..invalid sequence: tasks group set size ({n_groups}) != n_denses set size ({n_denses})!')
            return False
        
        n_different_tasks = len(set([x for x in decoded_seq.values() if type(x) == type(tuple)]))
        if n_different_tasks != self.n_tasks:
            print(f' ..invalid sequence: architecture does not contain all tasks: {n_different_tasks} != {self.n_tasks}!')
            return False
        
        print('  ..valid sequence!')
        return True
    

    def __check_sequence_validity(self, sequence):
        print(f'Sequence: {sequence}')
        decoded_seq = self.decode_sequence(sequence)
        print(f' .Decoded seq: {decoded_seq}')
        if len(set(decoded_seq.keys())) < 4:
            print(' ..invalid sequence: less than 4 task groups!')
            return False
        print('  ..valid sequence!')
        return True


    def __get_optimizer(self):
        if self.controller_optimizer.name == Optimizer.SGD.name:
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        elif self.controller_optimizer.name == Optimizer.SGD_NESTEROV.name:
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, nesterov=True, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer.value)(learning_rate=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        return optim


    def __create_control_model(self, controller_input_shape):
        main_input = Input(shape=controller_input_shape, name='main_input')        
        print(f'Controller model input shape: {main_input.shape}')
        x = RNN(LSTMCell(self.controller_lstm_dim), return_sequences=True)(main_input)
        #main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        main_output = Dense(self.search_space_size, activation='softmax', name='main_output')(x)
        print(f'Controller model output shape: {main_output.shape}')
        model = Model(inputs=[main_input], outputs=[main_output])
        return model


    def __train_control_model(self, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
        optim = self.__get_optimizer()
        
        self.controller_model.compile(optimizer=optim, loss={'main_output': loss_func})
        
        if os.path.exists(self.controller_weights_path):
            self.controller_model.load_weights(self.controller_weights_path)
        
        print("TRAINING CONTROLLER...")
        
        self.controller_model.fit({'main_input': x_data},
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        
        self.controller_model.save_weights(self.controller_weights_path)


    # ------------------- Hybrid Model -------------------

    
    def __create_hybrid_control_model(self, controller_input_shape, controller_batch_size):
        main_input = Input(shape=controller_input_shape, name='main_input')
        #x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        x = RNN(LSTMCell(self.controller_lstm_dim), return_sequences=True)(main_input)
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
        return model


    def __train_hybrid_control_model(self, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):

        optim = self.__get_optimizer()
        
        self.controller_model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                      loss_weights={'main_output': 1, 'predictor_output': 1})
        
        if os.path.exists(self.controller_weights_path):
            self.controller_model.load_weights(self.controller_weights_path)
        
        print("TRAINING CONTROLLER...")
        
        self.controller_model.fit({'main_input': x_data},
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
                   'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        
        self.controller_model.save_weights(self.controller_weights_path)

    
    def get_predicted_accuracies_hybrid_model(self, model, seqs):
        pred_accuracies = []
        for seq in seqs:
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies