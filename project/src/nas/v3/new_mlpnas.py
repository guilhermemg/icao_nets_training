import pickle
import numpy as np

import pyglove as pg

import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from src.nas.v3.nas_controller_4 import NASController_4
from src.nas.v3.mlp_search_space import New_MLPSearchSpace
from src.nas.v3.nas_algorithm import NASAlgorithm
from src.nas.v3.utils import clean_log, log_event

from src.configs.conf_interp import ConfigInterpreter
from src.m_utils.neptune_utils import NeptuneUtils

from src.base.experiment.training.model_trainer import ModelTrainer
from src.base.experiment.evaluation.model_evaluator import ModelEvaluator, DataSource


class New_MLPNAS(NASController_4):

    def __init__(self, train_gen, validation_gen, config_interp, neptune_utils):

        self.train_gen : ImageDataGenerator = train_gen
        self.validation_gen : ImageDataGenerator = validation_gen

        self.config_interp : ConfigInterpreter = config_interp
        self.neptune_utils : NeptuneUtils = neptune_utils

        self.controller_sampling_epochs   = self.config_interp.nas_params['controller_sampling_epochs']
        self.samples_per_controller_epoch = self.config_interp.nas_params['samples_per_controller_epochs']
        self.controller_train_epochs      = self.config_interp.nas_params['controller_training_epochs']
        self.architecture_train_epochs    = self.config_interp.nas_params['architecture_training_epochs']
        self.controller_loss_alpha        = self.config_interp.nas_params['controller_loss_alpha']

        self.search_space : New_MLPSearchSpace = New_MLPSearchSpace()
        self.nas_algorithm : NASAlgorithm = NASAlgorithm(self.config_interp.nas_params['nas_algorithm'])

        self.data : list = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        clean_log()

        super().__init__(config_interp)

        self.model_trainer = ModelTrainer(config_interp, neptune_utils)
        self.model_evaluator = ModelEvaluator(config_interp, neptune_utils)

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, self.config_interp.mlp_params['max_architecture_length'] - 1)
        self.controller_use_predictor = self.config_interp.controller_params['controller_use_predictor']
        
        if not self.controller_use_predictor:
            self.controller_model = self.create_control_model(self.controller_input_shape)
        else:
            self.controller_model = self.create_hybrid_control_model(self.controller_input_shape, self.controller_batch_size)


    def create_architecture(self, model_spec):
        self.model_trainer.create_model(self.train_gen, model_spec, running_nas=True)


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
            self.train_hybrid_control_model(model,
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
        search_space = self.search_space.get_search_space('ss_1')
        algo = self.nas_algorithm.get_algorithm()
        for model,feedback in pg.sample(search_space, algo, num_examples=self.samples_per_controller_epoch):
        #for controller_epoch in range(self.controller_sampling_epochs):
            print(70*'=')
            print(f'  New Controller Epoch | Feedback ID: {feedback.id} | Feedback DNA: {feedback.dna}')
            print(70*'-')
            #sequences = self.sample_architecture_sequences(self.controller_model, number_of_samples=self.samples_per_controller_epoch)
            
            #if self.use_predictor:
            #    pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            #    print('Predicted accuracies: ', pred_accuracies)
            
            #for i, sequence in enumerate(sequences):
            model_spec = model()
            #decoded_arch_seq = self.decode_sequence(sequence)
            
            print(f' -- Architecture {feedback.id}: {model_spec}')
            
            self.create_architecture(model_spec)
            self.train_architecture()
            final_eval = self.evaluate_architecture()
            
            # if self.use_predictor:
            #    self.append_model_metrics(model_spec, final_eval, pred_accuracies[i])
            # else:
            self.append_model_metrics(model_spec, final_eval)
            print(70*'=')
            
            #xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            
            #self.train_controller(self.controller_model, xc, yc, val_acc_target[-self.samples_per_controller_epoch:])

            feedback(final_eval['final_ACC'])
        
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        
        log_event()

        return self.data