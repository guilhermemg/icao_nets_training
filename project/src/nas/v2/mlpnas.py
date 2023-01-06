import pickle
import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.nas.v2.nas_controller_3 import NASController_3
from src.nas.v2.utils import *

from src.base.experiment.training.model_trainer import ModelTrainer
from src.base.experiment.evaluation.model_evaluator import ModelEvaluator, DataSource

class MLPNAS(NASController_3):

    def __init__(self, train_gen, validation_gen, config_interp, neptune_utils):

        self.train_gen = train_gen
        self.validation_gen = validation_gen

        self.config_interp = config_interp
        self.neptune_utils = neptune_utils

        self.controller_sampling_epochs   = self.config_interp.nas_params['controller_sampling_epochs']
        self.samples_per_controller_epoch = self.config_interp.nas_params['samples_per_controller_epochs']
        self.controller_train_epochs      = self.config_interp.nas_params['controller_training_epochs']
        self.architecture_train_epochs    = self.config_interp.nas_params['architecture_training_epochs']
        self.controller_loss_alpha        = self.config_interp.nas_params['controller_loss_alpha']

        self.data = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        clean_log()

        super().__init__(config_interp)

        self.model_trainer = ModelTrainer(config_interp, neptune_utils)
        self.model_evaluator = ModelEvaluator(config_interp, neptune_utils)

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, self.config_interp.mlp_params['max_architecture_length'] - 1)
        self.controller_model = self.create_control_model(self.controller_input_shape)


    def create_architecture(self, sequence):
        self.model_trainer.create_model(self.train_gen, sequence, running_nas=True)


    def train_architecture(self):
        self.model_trainer.train_model(self.train_gen, self.validation_gen, n_epochs=self.architecture_train_epochs, running_nas=True, fine_tuned=False)


    def evaluate_architecture(self):
        self.model_trainer.load_best_model()
        self.model_evaluator.set_data_src(DataSource.VALIDATION)
        final_eval = self.model_evaluator.test_model(self.validation_gen, self.model_trainer.model, verbose=False, running_nas=True)
        return final_eval


    def append_model_metrics_BAK(self, sequence, history, pred_accuracy=None):
        if len(history.history['val_accuracy']) == 1:
            if pred_accuracy:
                self.data.append([sequence,
                                  history.history['val_accuracy'][0],
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  history.history['val_accuracy'][0]])
            print('validation accuracy: ', history.history['val_accuracy'][0])
        else:
            val_acc = np.ma.average(history.history['val_accuracy'],
                                    weights=np.arange(1, len(history.history['val_accuracy']) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                  val_acc,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  val_acc])
            print('validation accuracy: ', val_acc)

    
    def append_model_metrics(self, sequence, final_eval, pred_accuracy=None):
        if pred_accuracy:
            self.data.append([sequence, final_eval['final_ACC'], pred_accuracy])
        else:
            self.data.append([sequence, final_eval['final_ACC']])

        print('validation accuracy: ', final_eval['final_ACC'])


    def prepare_controller_data(self, sequences):
        print('Preparing controller data...')
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]
        print(f'xc.shape: {xc.shape}')
        print(f'yc.shape: {yc.shape}')
        print(f'val_acc_target.shape: {len(val_acc_target)}')
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
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        
        # get discounted reward
        discounted_reward = self.get_discounted_reward(reward)

        # multiply discounted reward by log likelihood of actions to get loss function
        loss = - K.log(output) * discounted_reward[:, None]

        return loss


    def train_controller(self, model, x, y, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model,
                                    x,
                                    y,
                                    pred_accuracy,
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model,
                                     x,
                                     y,
                                     self.custom_loss,
                                     len(self.data),
                                     self.controller_train_epochs)


    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            for i, sequence in enumerate(sequences):
                decoded_arch_seq = self.decode_sequence(sequence)
                print(f' -- Architecture Number {i}: {decoded_arch_seq}')
                print(f' -- Sequence: {sequence}')
                self.create_architecture(decoded_arch_seq)
                self.train_architecture()
                final_eval = self.evaluate_architecture()
                if self.use_predictor:
                    self.append_model_metrics(sequence, final_eval, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, final_eval)
                print('------------------------------------------------------')
            
            xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            
            self.train_controller(self.controller_model, xc, yc, val_acc_target[-self.samples_per_controller_epoch:])
        
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        
        log_event()
        
        return self.data


    def __sort_search_data(self, nas_data):
        val_accs = [item[1] for item in nas_data]
        sorted_idx = np.argsort(val_accs)[::-1]
        nas_data = [nas_data[x] for x in sorted_idx]
        return nas_data

    
    def get_top_n_architectures(self, top_n=5):
        data = self.__sort_search_data(self.data)
        search_space = MLPSearchSpace(self.dataset, self.min_task_group_size)
        print('Top {} Architectures:'.format(top_n))
        for seq_data in data[:top_n]:
            print('Architecture', search_space.decode_sequence(seq_data[0]))
            print('Validation Accuracy:', seq_data[1])
