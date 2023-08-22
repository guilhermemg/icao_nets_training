import os
import time
import neptune
import nats_bench

from copy import deepcopy

import pyglove as pg
import pandas as pd

from validation_src.val_rl_dna_generator import RL_DNAGenerator
from validation_src.val_config_interp import ConfigInterpreter
from validation_src.val_config_args import kwargs, NEPTUNE_API_TOKEN, NEPTUNE_PROJECT


DEFAULT_NATS_FILES = dict(tss=None, sss=None)
DEFAULT_REPORTING_EPOCH = dict(tss=200, sss=90)
VALIDATION_SET_REPORTING_EPOCH = 12


@pg.functor([('channels', pg.typing.List(pg.typing.Int()))])
def model_sss_spc(channels):
    return ':'.join(str(x) for x in channels)


@pg.functor([('ops', pg.typing.List(pg.typing.Str())),('num_nodes', pg.typing.Int())])
def model_tss_spc(ops, num_nodes):
    """The architecture in the topology search space of NATS-Bench."""
    nodes, k = [], 0
    for i in range(1, num_nodes):
        xstrs = []
        for j in range(i):
            xstrs.append('{:}~{:}'.format(ops[k], j))
            k += 1
        nodes.append('|' + '|'.join(xstrs) + '|')
    return '+'.join(nodes)



class NASExecutor:
    def __init__(self, algorithm_name, dataset_name, max_train_hours, ss_indicator, use_neptune):
        print(80*'-')
        print(f'Preparing NASExecutor:')
        print(f'  Algorithm: {algorithm_name}')
        print(f'  Dataset: {dataset_name}')
        print(f'  Max Training Hours: {max_train_hours}')
        print(f'  Search Space Indicator: {ss_indicator}')
        print(f'  Use Neptune: {use_neptune}')
        
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.ss_indicator = ss_indicator
        self.max_train_hours = max_train_hours

        self.use_neptune = use_neptune
        if self.use_neptune:
            self.run = neptune.init_run(name=f'NAS with NATS and {self.ss_indicator.upper()}',
                                        tags=['nas', 'nats', self.ss_indicator, 'adamax'], 
                                        description=f'NAS with NATS and {self.ss_indicator.upper()} using ADAMAX as optimizer.',
                                        project=NEPTUNE_PROJECT, 
                                        api_token=NEPTUNE_API_TOKEN,
                                        source_files=['*.py'])

        self.config_interp = ConfigInterpreter(kwargs)

        self.nats_api = self.__create_nats_api()
        self.search_space = self.__get_search_space()
        self.reporting_epoch = self.__get_reporting_epoch()
        self.algorithm = self.__get_algorithm()


    def __create_nats_api(self):
        nats_bench.api_utils.reset_file_system('default')
        file_path_or_dict = DEFAULT_NATS_FILES[self.ss_indicator]
        return nats_bench.create(file_path_or_dict, self.ss_indicator, fast_mode=True, verbose=False)        


    def __get_reporting_epoch(self):
         return DEFAULT_REPORTING_EPOCH[self.ss_indicator]


    def __get_search_space(self):
        info = nats_bench.search_space_info('nats-bench', self.ss_indicator)
        if self.ss_indicator == 'sss':
            print(f'  .. Number of layers: {info["num_layers"]}')
            print(f'  .. Candidates: {info["candidates"]}')
            return model_sss_spc(pg.sublist_of(info['num_layers'], info['candidates'], choices_distinct=False))
        elif self.ss_indicator == 'tss':
            print(f'  .. Operations: {info["op_names"]}')
            print(f'  .. Nodes: {info["num_nodes"]}')
            total = info['num_nodes'] * (info['num_nodes'] - 1) // 2
            return model_tss_spc(pg.sublist_of(total, info['op_names'], choices_distinct=False), info['num_nodes'])


    def __get_algorithm(self):
        """Creates algorithm."""
        if self.algorithm_name == 'random':
            return pg.generators.Random()
        elif self.algorithm_name == 'evolution':
            return pg.evolution.regularized_evolution(mutator=pg.evolution.mutators.Uniform(), population_size=50, tournament_size=10)
        elif self.algorithm_name == 'rl':
            return RL_DNAGenerator(self.config_interp)
        else:
            return pg.load(self.algorithm_name)


    def __search(self, nats_api, search_model, algo, dataset='cifar10', reporting_epoch=12, max_train_hours=2e4):
        nats_api.reset_time()
        valid_models = 0
        time_spent_in_secs = 0
        start_time = time.time()
        last_report_time = start_time

        results_df = pd.DataFrame(columns=['id','dna','cell_spec',
                                           'val_acc','pred_acc','latency','time_cost','total_time',
                                           'test_acc','test_loss','test_per_time','test_all_time',
                                           'time_spent_in_hours','time_spent_in_secs',
                                           'train_accuracy','train_loss','train_per_time','train_all_time',
                                           'comment'])

        for model, feedback in pg.sample(search_model, algo):
            spec = model()

            (validation_accuracy, latency, time_cost, total_time) = nats_api.simulate_train_eval(spec, dataset=dataset, hp=VALIDATION_SET_REPORTING_EPOCH)

            time_spent_in_secs = nats_api.used_time

            more_info = nats_api.get_more_info(spec, dataset, hp=reporting_epoch)  # pytype: disable=wrong-arg-types  # dict-kwargs

            valid_models += 1

            feedback(validation_accuracy)

            time_spent_in_hours = time_spent_in_secs / (60 * 60)

            if time_spent_in_hours > max_train_hours:
                break # Break the first time we exceed the budget.

            if feedback.id % 100 == 0:
                now = time.time()
                print(f'Tried {feedback.id} models, valid {valid_models}, '
                      f'time_spent_in_hours: {int(time_spent_in_hours)}h, '
                      f'time_spent_in_secs: {round(time_spent_in_secs,3)}s, '
                      f'elapse since last report: {round(now - last_report_time,3)}s.')
                last_report_time = now

            if self.ss_indicator == 'sss':
                formatted_spec = ':'.join(['{:02d}'.format(int(x)) for x in spec.split(':')])
            elif self.ss_indicator == 'tss':
                formatted_spec = spec

            pred_acc = -1
            if self.algorithm_name == 'rl':
                if self.config_interp.controller_params['controller_use_predictor']:
                    df = pd.read_csv('./LOGS/nas_data.csv')
                    pred_acc = df['pred_acc'].values[-1]

            #print(f'Cell-spec: {formatted_spec} | ID: {feedback.id} | DNA: {feedback.dna} | Validation Acc: {validation_accuracy}')

            results_df.loc[len(results_df)] = {'id': feedback.id,
                                               'cell_spec': formatted_spec, 
                                               'dna': feedback.dna, 
                                               'val_acc': validation_accuracy, 
                                               'pred_acc': pred_acc,
                                               'latency': latency,
                                               'time_cost': time_cost,
                                               'total_time': total_time,
                                               'test_acc': more_info['test-accuracy'],
                                               'test_loss': more_info['test-loss'],
                                               'test_per_time': more_info['test-per-time'],
                                               'test_all_time': more_info['test-all-time'],
                                               'time_spent_in_secs': round(time_spent_in_secs,3),
                                               'time_spent_in_hours': int(time_spent_in_hours),
                                               'train_loss': more_info['train-loss'],
                                               'train_accuracy': more_info['train-accuracy'],
                                               'train_per_time': more_info['train-per-time'],
                                               'train_all_time': more_info['train-all-time'],
                                               'comment': more_info['comment'] if 'comment' in more_info else ''}


        print(f'Total time elapse: {time.time() - start_time} seconds.')

        return results_df


    def test_nas_algo(self, dest_path=None):
        results_df = self.__search(self.nats_api, self.search_space, self.algorithm, self.dataset_name, self.reporting_epoch, self.max_train_hours)

        sorted_results = results_df.sort_values(by='val_acc', ascending=False)

        sorted_results['algorithm'] = self.algorithm
        sorted_results['dataset'] = self.dataset_name
        sorted_results['max_train_hours'] = self.max_train_hours

        self.print_report(sorted_results)
        
        dest_path = f'./data/pred_acc/{self.ss_indicator}/{self.algorithm_name}_{str(self.max_train_hours)}h_{self.dataset_name}.csv' if dest_path is None else dest_path
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        self.save_report(sorted_results, dest_path)

        if self.use_neptune:
            self.log_data_to_neptune(sorted_results)

        print('Search completed.')
        print('-'*80)

        return sorted_results


    def print_report(self, sorted_results_df):
        row_0 = sorted_results_df.iloc[0]
        print('-'*80)
        print(f'Total evaluated architectures: {len(sorted_results_df)}')
        print(f'Total time spent:              {row_0["time_spent_in_hours"]} hours')
        print(f'Best model found:              {row_0["cell_spec"]}')
        print(f'Best model DNA:                {row_0["dna"]}')
        print(f'Validation accuracy:           {row_0["val_acc"]}')
        print(f'Test accuracy:                 {row_0["test_acc"]}')
        print('-'*80)

        #display(sorted_results_df.iloc[:,:13].head(1))


    def save_report(self, sorted_results_df, filename):
        sorted_results_df.to_csv(filename, index=False)    

    
    def __log_best_arch(self, sorted_results_df, run):
        first_row = sorted_results_df.iloc[0]

        run['best_arch/val_acc']             =  first_row['val_acc']
        run['best_arch/test_acc']            =  first_row['test_acc']
        run['best_arch/test_loss']           =  first_row['test_loss']
        run['best_arch/test_per_time']       =  first_row['test_per_time']
        run['best_arch/test_all_time']       =  first_row['test_all_time']
        run['best_arch/train_loss']          =  first_row['train_loss']
        run['best_arch/train_accuracy']      =  first_row['train_accuracy']
        run['best_arch/train_per_time']      =  first_row['train_per_time']
        run['best_arch/train_all_time']      =  first_row['train_all_time']
        run['best_arch/time_spent_in_secs']  =  first_row['time_spent_in_secs']
        run['best_arch/time_spent_in_hours'] =  first_row['time_spent_in_hours']
        run['best_arch/max_train_hours']     =  first_row['max_train_hours']
        run['best_arch/id']                  =  first_row['id']
        run['best_arch/latency']             =  first_row['latency']
        run['best_arch/time_cost']           =  first_row['time_cost']
        run['best_arch/total_time']          =  first_row['total_time']
        run['best_arch/cell_spec']           =  first_row['cell_spec']
        run['best_arch/dna']                 =  str(first_row['dna'])
        run['best_arch/comment']             =  first_row['comment']
        run['best_arch/algorithm']           =  str(first_row['algorithm'])
        run['best_arch/dataset']             =  str(first_row['dataset'])


    def __log_nas_history(self, sorted_results_df, run):
        history_data = sorted_results_df.sort_values(by='id', ascending=True)

        cols = ['val_acc','test_acc','test_loss','test_per_time','test_all_time','train_loss','train_accuracy','train_per_time','train_all_time',
                'time_spent_in_secs','time_spent_in_hours','total_time','time_cost','latency']
        
        for col in cols:
            for v in history_data[col].values:
                run['nas_history/'+col].append(v)


    def log_data_to_neptune(self, sorted_results_df):
        self.run['params'] = {'dataset': self.dataset_name,
                         'max_train_hours': self.max_train_hours,
                         'algorithm': self.algorithm_name,
                         'search_space': self.ss_indicator}
        
        if self.algorithm_name == 'rl':
            ctr_params = deepcopy(kwargs['controller_params'])
            ctr_params['controller_optimizer'] = str(ctr_params['controller_optimizer'])
            self.run['controller_params'] = ctr_params
        
        self.__log_best_arch(sorted_results_df, self.run)
        self.__log_nas_history(sorted_results_df, self.run)

        self.run.stop()
        