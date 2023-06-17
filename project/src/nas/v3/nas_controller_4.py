import os
import numpy as np

from itertools import product

import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences


from src.base.experiment.training.optimizers import Optimizer
from src.nas.v3.mlp_search_space import MLPSearchSpaceCandidate

class NASController_4:
    def __init__(self, config_interp):        
        self.config_interp = config_interp

        #self.max_len                    = self.config_interp.mlp_params['max_architecture_length']
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
        
        self.controller_classes = self.config_interp.nas_params['nas_search_space'].value['n_classes']

        self.controller_noise_dim = 4
        self.controller_input_shape = (1, self.controller_classes+self.controller_noise_dim)
        
        self.controller_batch_size = self.config_interp.controller_params['controller_batch_size']

        self.controller_use_predictor = self.config_interp.controller_params['controller_use_predictor']
        
        self.nas_data_history = None
        self.search_space_candidates = MLPSearchSpaceCandidate.N_DENSES.value + MLPSearchSpaceCandidate.N_CONVS.value
        self.search_space_candidates_size = len(self.search_space_candidates)

        self.__clean_controller_weights()        

        if not self.controller_use_predictor:
            self.controller_model = self.__create_control_model()
        else:
            self.controller_model = self.__create_hybrid_control_model()


    def __clean_controller_weights(self):
        if os.path.exists(self.controller_weights_path):
            os.remove(self.controller_weights_path)


    def build_new_arch(self, prev_arch):
        inp = np.array(prev_arch).reshape(1,1,self.controller_classes)

        print(f'input: {inp}')

        final_arch = []
        for i in range(self.controller_classes):
            noise = np.random.randint(0, 10, size=4).reshape(1,1,4)
            print(f'noise: {noise}')

            inp = np.concatenate([inp,noise], axis=2)

            prob_list = self.controller_model.predict(inp)   # output: (1,1,n_classes+noise_dim)
            print(f'prob_list.shape: {prob_list.shape}')
            prob_list = prob_list[0][0]
            print(f'prob_list: {prob_list}')
            print(f'len(prob_list): {len(prob_list)}')

            lookup_search_space_candidates = None
            if i < 4:   # first half of DNA
                print(f'prob_list[:5]: {prob_list[:5]}')
                print(f'len(prob_list[:5]): {len(prob_list[:5])}')
                chose_idx = np.argmax(prob_list[:5])  # choose from the first part of the list (5 classes)
                lookup_search_space_candidates = self.search_space_candidates[:5]
            elif i >= 4:  # second half of DNA
                print(f'prob_list[5:]: {prob_list[5:]}')
                print(f'len(prob_list[5:]): {len(prob_list[5:])}')
                chose_idx = np.argmax(prob_list[5:])  # choose from the second part of the list (3 classes)
                lookup_search_space_candidates = self.search_space_candidates[5:]
            
            print(f'chose_idx: {chose_idx} | lookup_search_space_candidates[chose_idx]: {lookup_search_space_candidates[chose_idx]} | lookup_search_space_candidates: {lookup_search_space_candidates}')

            final_arch.append(lookup_search_space_candidates[chose_idx])
            print(f'final_arch: {final_arch}')

            new_arch = pad_sequences([final_arch], maxlen=self.controller_classes, padding='post', value=-1)
            print(new_arch.shape)

            inp = np.array(new_arch).reshape(1,1,self.controller_classes)
            print(f'new input: {inp}')
        
        final_arch = [[final_arch[:4], [final_arch[4:]]]]
        print(f'final_arch: {final_arch}')

        return final_arch


    def __prepare_controller_data(self, nas_data_history):
        self.nas_data_history = nas_data_history

        print('Preparing controller data...')
        print(f' ..nas_data_history: {nas_data_history}')
        
        xc = np.array([[item[0].to_numbers() for item in nas_data_history]])       
        xc = xc.reshape(self.controller_batch_size, 1, self.controller_classes)
        print(f' ..xc.shape: {xc.shape}')

        noise = np.random.randint(0, 10, size=(self.controller_batch_size, 1, self.controller_noise_dim))
        xc = np.concatenate([xc, noise], axis=2)
        print(f' ..xc + noise.shape: {xc.shape}')

        #yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        yc = np.zeros((self.controller_batch_size, 1, self.controller_classes))
        print(f' ..yc.shape: {yc.shape}')
        
        val_acc_target = [item[1] for item in nas_data_history]
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


    def __get_optimizer(self):
        if self.controller_optimizer.name == Optimizer.SGD.name:
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        elif self.controller_optimizer.name == Optimizer.SGD_NESTEROV.name:
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, nesterov=True, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer.value)(learning_rate=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        return optim


    def __create_control_model(self):
        main_input = Input(shape=self.controller_input_shape, name='main_input')        
        print(f'Controller model input shape: {main_input.shape}')
        x = RNN(LSTMCell(self.controller_lstm_dim), return_sequences=True)(main_input)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
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

    
    def __create_hybrid_control_model(self):
        main_input = Input(shape=self.controller_input_shape, name='main_input')
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
            control_sequences = pad_sequences([seq], maxlen=self.controller_classes, padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.controller_classes - 1)
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies