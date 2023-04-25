import itertools


class MLPSearchSpace(object):

    def __init__(self, dataset, min_task_group_size):
        self.dataset = dataset
        self.tasks = self.dataset.value['tasks']
        
        self.min_task_group_size = min_task_group_size

        self.vocab = self.vocab_dict()

        print(f'Vocab: {self.vocab}')
    

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


    def vocab_dict_BAK(self):
        def subsets(nums):
            result = []
            for i in range(len(nums) + 1):
                result += itertools.combinations(nums, i)
            return result

        list_n_fcs = [1,2,3,4,5]
        tasks = self.tasks
        tasks_groups_list = subsets(tasks)
        tasks_groups_list = [x for x in tasks_groups_list if len(x) >= self.min_task_group_size]
        
        tasks_groups_params = []
        tasks_groups_id = []
        
        for i in range(len(list_n_fcs)):
            for j in range(len(tasks_groups_list)):
                tasks_groups_params.append((f'g{j}', tasks_groups_list[j]))
                tasks_groups_params.append((f'n_denses_{j}', list_n_fcs[i]))
                tasks_groups_id.append(len(tasks_groups_list) * i + j + 1)
        
        vocab = dict(zip(tasks_groups_id, tasks_groups_params))
        
        print(f'Vocab Size: {len(vocab)}')

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