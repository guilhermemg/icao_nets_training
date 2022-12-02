import os
import pickle
import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, LSTMCell, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.nas.v2.mlp_search_space import MLPSearchSpace


class NASController_3(MLPSearchSpace):
    def __init__(self, config_interp):        
        self.config_interp = config_interp

        self.max_len                = self.config_interp.mlp_params['max_architecture_length']
        self.controller_lstm_dim    = self.config_interp.controller_params['controller_lstm_dim']
        self.controller_optimizer   = self.config_interp.controller_params['controller_optimizer']
        self.controller_lr          = self.config_interp.controller_params['controller_learning_rate']
        self.controller_decay       = self.config_interp.controller_params['controller_decay']
        self.controller_momentum    = self.config_interp.controller_params['controller_momentum']
        self.use_predictor          = self.config_interp.controller_params['controller_use_predictor']

        self.controller_weights_path = 'LOGS/controller_weights.h5'

        self.seq_data = []

        n_tasks = len(self.config_interp.prop_args['benchmarking']['dataset'].value['tasks'])

        super().__init__(n_tasks)

        self.controller_classes = len(self.vocab) + 1

   
    def sample_architecture_sequences_BAK(self, model, number_of_samples):
        final_layer_id = len(self.vocab)
        dropout_id = final_layer_id - 1
        vocab_idx = [0] + list(self.vocab.keys())
        samples = []
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')
        while len(samples) < number_of_samples:
            seed = []
            while len(seed) < self.max_len:
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len - 1)
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                probab = probab[0][0]
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]
                if next == dropout_id and len(seed) == 0:
                    continue
                if next == final_layer_id and len(seed) == 0:
                    continue
                if next == final_layer_id:
                    seed.append(next)
                    break
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break
                if not next == 0:
                    seed.append(next)
            if seed not in self.seq_data:
                samples.append(seed)
                self.seq_data.append(seed)
        return samples


    # def sample_architecture_sequences(self, model, number_of_samples):
    #     final_layer_id = len(self.vocab)
    #     vocab_idx = [0] + list(self.vocab.keys())
    #     samples = []
    #     print("GENERATING ARCHITECTURE SAMPLES...")
    #     print('------------------------------------------------------')
    #     while len(samples) < number_of_samples:
    #         seed = []
    #         while len(seed) < self.max_len:
    #             sequence = pad_sequences([seed], maxlen=self.max_len-1, padding='post')
    #             sequence = sequence.reshape(1, 1, self.max_len-1)
    #             if self.use_predictor:
    #                 (probab, _) = model.predict(sequence)
    #             else:
    #                 probab = model.predict(sequence)
        #         probab = probab[0][0]
        #         next = np.random.choice(vocab_idx, size=1, p=probab)[0]
        #         if next == final_layer_id and len(seed) == 0:
        #             continue
        #         if next == final_layer_id and len(seed) <= self.max_len:
        #             continue
        #         if next == final_layer_id:
        #             seed.append(next)
        #             break
        #         if len(seed) == self.max_len - 1:
        #             seed.append(final_layer_id)
        #             break
        #         if not next == 0:
        #             seed.append(next)
        #     if seed not in self.seq_data:
        #         samples.append(seed)
        #         self.seq_data.append(seed)
        # return samples

    
    def __check_sequence_validity(self, sequence):
        decoded_seq = self.decode_sequence(sequence)
        if len(set(decoded_seq.keys())) < 4:
            return False
        return True


    def sample_architecture_sequences(self, model, number_of_samples):
        final_layer_id = len(self.vocab)
        vocab_idx = [0] + list(self.vocab.keys())
        samples = []
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')
        print(f'Number of samples: {number_of_samples}')
        while len(samples) < number_of_samples:
            seed = []
            while len(seed) < self.max_len:
                sequence = pad_sequences([seed], maxlen=self.max_len-1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len-1)
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                
                probab = probab[0][0]
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]
                if next == final_layer_id and len(seed) == 0:
                    continue
                if next == final_layer_id and len(seed) <= self.max_len:
                    continue
                if next == final_layer_id:
                    seed.append(next)
                    break
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break
                if not next == 0:
                    seed.append(next)

            if seed not in self.seq_data:
                if self.__check_sequence_validity(seed):
                    samples.append(seed)
                    self.seq_data.append(seed)

        return samples


    def create_control_model(self, controller_input_shape):
        main_input = Input(shape=controller_input_shape, name='main_input')        
        x = RNN(LSTMCell(self.controller_lstm_dim), return_sequences=True)(main_input)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output])
        return model


    def train_control_model(self, model, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        
        model.compile(optimizer=optim, loss={'main_output': loss_func})
        
        if os.path.exists(self.controller_weights_path):
            model.load_weights(self.controller_weights_path)
        
        print("TRAINING CONTROLLER...")
        
        model.fit({'main_input': x_data},
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        
        model.save_weights(self.controller_weights_path)
