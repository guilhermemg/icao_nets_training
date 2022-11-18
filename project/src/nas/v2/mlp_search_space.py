

class MLPSearchSpace(object):

    def __init__(self, target_classes):
        self.target_classes = target_classes
        self.vocab = self.vocab_dict()


    def vocab_dict_BAK(self):
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        layer_params = []
        layer_id = []
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)
        vocab = dict(zip(layer_id, layer_params))
        vocab[len(vocab) + 1] = (('dropout'))
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        return vocab
    

    def vocab_dict(self):
        list_n_fcs = [1,2,3,4,5]
        layers = ['n_denses_0','n_denses_1','n_denses_2','n_denses_3']
        layers_params = []
        layer_id = []
        for i in range(len(list_n_fcs)):
            for j in range(len(layers)):
                layers_params.append((layers[j], list_n_fcs[i]))
                layer_id.append(len(layers) * i + j + 1)
        vocab = dict(zip(layer_id, layers_params))
        return vocab



    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence


    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        decoded_sequence = {x:y for x,y in decoded_sequence}
        return decoded_sequence