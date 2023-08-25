import os
import pandas as pd
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import normalize

import matplotlib.pyplot as plt

from src.base.experiment.training.optimizers import Optimizer
from src.nas.v3.mlp_search_space import MLPSearchSpaceCandidate

class NASController_4:
    def __init__(self, config_interp):        
        self.config_interp = config_interp
        
        self.controller_classes               = self.config_interp.controller_params['controller_classes']
        self.controller_lstm_dim              = self.config_interp.controller_params['controller_lstm_dim']
        self.controller_optimizer             = self.config_interp.controller_params['controller_optimizer']
        self.controller_lr                    = self.config_interp.controller_params['controller_learning_rate']
        self.controller_decay                 = self.config_interp.controller_params['controller_decay']
        self.controller_momentum              = self.config_interp.controller_params['controller_momentum']
        self.controller_loss_alpha            = self.config_interp.controller_params['controller_loss_alpha']
        self.controller_train_epochs          = self.config_interp.controller_params['controller_training_epochs']
        self.controller_max_proposed_arch_len = self.config_interp.controller_params['controller_max_proposed_arch_len']
        self.controller_batch_size            = self.config_interp.controller_params['controller_batch_size']

        self.controller_weights_path = 'LOGS/controller_weights.h5'
        self.controller_losses_path = 'LOGS/controller_losses.csv'

        # self.n_tasks = len(self.config_interp.tasks)
        
        self.controller_noise_dim = 4
        
        self.controller_model_input_shape = (1, self.controller_max_proposed_arch_len+self.controller_noise_dim) # (1,5+4)
        self.controller_model_output_shape = (1, self.controller_classes)  # (1,8)
        
        self.nas_data_history = None
        #self.search_space_candidates = MLPSearchSpaceCandidate.N_DENSES.value + MLPSearchSpaceCandidate.N_CONVS.value
        #self.search_space_candidates_size = len(self.search_space_candidates)

        self.__clean_controller_losses()
        self.__clean_controller_weights()        

        self.controller_use_predictor = self.config_interp.controller_params['controller_use_predictor']
        if not self.controller_use_predictor:
            self.controller_model = self.__create_control_model()
        else:
            self.controller_model = self.__create_hybrid_control_model()


    def __clean_controller_losses(self):
        if os.path.exists(self.controller_losses_path):
            os.remove(self.controller_losses_path)


    def __clean_controller_weights(self):
        if os.path.exists(self.controller_weights_path):
            os.remove(self.controller_weights_path)
    
            
    def build_new_arch(self, prev_arch):
        inp = np.array(prev_arch).reshape(1,1,self.controller_max_proposed_arch_len)

        # print(f' ..input: {inp}')

        final_arch = []
        for i in range(self.controller_max_proposed_arch_len):
            noise = np.random.randint(0, 10, size=self.controller_noise_dim).reshape(1,1,self.controller_noise_dim)
            # print(f' ..noise: {noise}')

            inp = np.concatenate([inp,noise], axis=2)
            # print(inp.shape)

            prob_list, pred_acc = None, None
            if self.controller_use_predictor:
                prob_list, pred_acc = self.controller_model.predict(inp)   # output: (1,1,controller_max_proposed_arch_len+noise_dim), pred_accuracy
                pred_acc = pred_acc[0][0][0]*100
            else:
                prob_list = self.controller_model.predict(inp)   # output: (1,1,controller_max_proposed_arch_len+noise_dim)

            prob_list = prob_list[0][0]

            # print(f'prob_list: {prob_list}')
            # print(f'pred_acc: {pred_acc}')

            chose_idx = np.argmax(prob_list)  # choose from the first part of the list (controller_max_proposed_arch_len)
            
            final_arch.append(int(chose_idx))
            # print(f'final_arch: {final_arch}')

            new_arch = pad_sequences([final_arch], maxlen=self.controller_max_proposed_arch_len, padding='post', value=-1)
            # print(new_arch.shape)

            inp = np.array(new_arch).reshape(1,1,self.controller_max_proposed_arch_len)
            # print(f' ..new input: {inp}')
        
        final_arch = [[final_arch]]
        # print(f' .final_arch: {final_arch}')
        
        return final_arch, pred_acc


    def __prepare_controller_data(self, nas_data_history):
        # print('Preparing controller data...')
        
        self.nas_data_history = nas_data_history
        # print(f' ..len(nas_data_history): {len(nas_data_history)}')

        last_nas_data_history_batch = self.nas_data_history[-self.controller_batch_size:] 
        # print(f'len(last_nas_data_history_batch): {len(last_nas_data_history_batch)}')
        # print(f' ..last_nas_data_history_batch[:10]: {last_nas_data_history_batch[:10]}')
        
        xc = np.array([[item[0].to_numbers() for item in last_nas_data_history_batch]])       
        xc = xc.reshape(self.controller_batch_size, 1, self.controller_max_proposed_arch_len)
        # print(f' ..xc.shape: {xc.shape}')

        noise = np.random.randint(0, 10, size=(self.controller_batch_size, 1, self.controller_noise_dim))
        xc = np.concatenate([xc, noise], axis=2)
        xc = normalize(xc, axis=2)
        # print(f' ..xc + noise.shape: {xc.shape}')
        # print(f' ..xc + noise[:10]: {xc[:10]}')

        #yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        yc = np.array([item[1].to_numbers() for item in last_nas_data_history_batch])
        # print(f' ..yc[:10]: {yc[:10]}')
        yc = normalize(yc, axis=1)
        # print(f' ..yc[:10]: {yc[:10]}')

        yc = yc.reshape(self.controller_batch_size, 1, self.controller_max_proposed_arch_len)
        #yc = np.zeros((self.controller_batch_size, 1, self.controller_max_proposed_arch_len))
        # print(f' ..yc.shape: {yc.shape}')
        
        val_acc_target = [item[2] for item in last_nas_data_history_batch]
        # print(f' ..val_acc_target[:10]: {val_acc_target[:10]}')  

        val_acc_target = normalize(val_acc_target, axis=0)
        # print(f' ..val_acc_target[:10]: {val_acc_target[:10]}')

        val_acc_target = np.array(val_acc_target).reshape(self.controller_batch_size, 1, 1)
        # print(f' ..val_acc_target.shape: {val_acc_target.shape}')
              
        # print(' ..done!')
        
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
        reward = np.array([item[2] - baseline for item in self.nas_data_history[-self.controller_batch_size:]]).reshape(
           self.controller_batch_size, 1)
        
        # get discounted reward
        discounted_reward = self.get_discounted_reward(reward)

        # multiply discounted reward by log likelihood of actions to get loss function
        loss = - K.log(output) * discounted_reward[:, None]

        return loss


    def train_model_controller(self, nas_data_history):
        # print(' ..Training controller model...')

        xc, yc, val_acc_target = self.__prepare_controller_data(nas_data_history)

        if self.controller_use_predictor:
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
        
        # print(' ..Controller model trained!')


    def __get_optimizer(self):
        if self.controller_optimizer.name == Optimizer.SGD.name:
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        elif self.controller_optimizer.name == Optimizer.SGD_NESTEROV.name:
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, nesterov=True, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer.value)(learning_rate=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        return optim


    def __create_control_model(self):
        main_input = Input(shape=self.controller_model_input_shape, name='main_input')        
        # print(f'Controller model input shape: {main_input.shape}')
        x = RNN(LSTMCell(self.controller_lstm_dim), return_sequences=True)(main_input)
        main_output = Dense(self.controller_model_output_shape[1], activation='softmax', name='main_output')(x)
        # print(f'Controller model output shape: {main_output.shape}')
        model = Model(inputs=[main_input], outputs=[main_output])
        return model


    def __draw_training_history(self):
        df = pd.read_csv(self.controller_losses_path)
        losses = df['loss']

        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.plot(range(len(losses)), losses, label='loss')
        
        for i in range(0, len(losses)+1, self.controller_train_epochs):
            plt.axvline(x=i, color='r', linestyle='--')

        plt.show()


    def __save_controller_losses(self, history):
        if os.path.exists(self.controller_losses_path):
            df = pd.read_csv(self.controller_losses_path)
        else:
            df = pd.DataFrame(columns=['loss'])

        aux_df = pd.DataFrame(history.history['loss'], columns=['loss'])

        final_df = pd.concat([df, aux_df], ignore_index=True)
        final_df.to_csv(self.controller_losses_path, index=False)

        # print(f'Losses saved into {self.controller_losses_path}!')


    def __train_control_model(self, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
        optim = self.__get_optimizer()
        
        self.controller_model.compile(optimizer=optim, loss={'main_output': loss_func})
        
        if os.path.exists(self.controller_weights_path):
            self.controller_model.load_weights(self.controller_weights_path)
        
        # print("TRAINING CONTROLLER...")
        
        H = self.controller_model.fit({'main_input': x_data},
                                      {'main_output': y_data.reshape(len(y_data), 1, self.controller_max_proposed_arch_len)},
                                       epochs=nb_epochs,
                                       batch_size=controller_batch_size,
                                       verbose=0)
        
        self.__save_controller_losses(H)

        self.__draw_training_history()

        self.controller_model.save_weights(self.controller_weights_path)


    # ------------------- Hybrid Model -------------------

    
    def __create_hybrid_control_model(self):
        main_input = Input(shape=self.controller_model_input_shape, name='main_input')
        x = RNN(LSTMCell(self.controller_lstm_dim), return_sequences=True)(main_input)
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
        main_output = Dense(self.controller_model_output_shape[1], activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
        return model


    def __train_hybrid_control_model(self, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):

        optim = self.__get_optimizer()
        
        self.controller_model.compile(optimizer=optim,
                                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                                      loss_weights={'main_output': 1, 'predictor_output': 1})
        
        if os.path.exists(self.controller_weights_path):
            self.controller_model.load_weights(self.controller_weights_path)
        
        # print("TRAINING CONTROLLER...")
                
        H = self.controller_model.fit({'main_input': x_data}, 
                                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_max_proposed_arch_len), 
                                   'predictor_output': pred_target}, 
                                   epochs=nb_epochs,
                                   batch_size=controller_batch_size,
                                   verbose=0)
        
        self.__save_controller_losses(H)

        self.__draw_training_history()

        self.controller_model.save_weights(self.controller_weights_path)

    
    def get_predicted_accuracies_hybrid_model(self, model, seqs):
        pred_accuracies = []
        for seq in seqs:
            control_sequences = pad_sequences([seq], maxlen=self.controller_model_input_shape[1], padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.controller_model_input_shape[1] - 1)
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies