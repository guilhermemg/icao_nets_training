import time
import nats_bench

import pyglove as pg
import pandas as pd

from validation_src.val_rl_dna_generator import RL_DNAGenerator
from validation_src.val_config_interp import ConfigInterpreter
from validation_src.val_config_args import kwargs


DEFAULT_NATS_FILEs = dict(tss=None, sss=None)
DEFAULT_REPORTING_EPOCH = dict(tss=200, sss=90)
VALIDATION_SET_REPORTING_EPOCH = 12


@pg.functor([('channels', pg.typing.List(pg.typing.Int()))])
def model_sss_spc(channels):
    return ':'.join(str(x) for x in channels)


class NASExecutor:
    def __get_search_space(self, ss_indicator):
        info = nats_bench.search_space_info('nats-bench', ss_indicator)
        print(f'Candidates: {info["candidates"]}')
        if ss_indicator == 'sss':
            return model_sss_spc(pg.sublist_of(info['num_layers'], info['candidates'], choices_distinct=False))


    def __get_algorithm(self, algorithm_str):
        """Creates algorithm."""
        if algorithm_str == 'random':
            return pg.generators.Random()
        elif algorithm_str == 'evolution':
            return pg.evolution.regularized_evolution(mutator=pg.evolution.mutators.Uniform(), population_size=50, tournament_size=10)
        elif algorithm_str == 'rl':
            config_interp = ConfigInterpreter(kwargs)
            return RL_DNAGenerator(config_interp)
        else:
            return pg.load(algorithm_str)


    def __search(self, nats_api, search_model, algo, dataset='cifar10', reporting_epoch=12, max_train_hours=2e4):
        nats_api.reset_time()
        valid_models = 0
        time_spent_in_secs = 0
        start_time = time.time()
        last_report_time = start_time

        results_df = pd.DataFrame(columns=['id','dna','cell_spec',
                                           'val_acc','latency','time_cost','total_time',
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

            formatted_spec = ':'.join(['{:02d}'.format(int(x)) for x in spec.split(':')])
            #print(f'Cell-spec: {formatted_spec} | ID: {feedback.id} | DNA: {feedback.dna} | Validation Acc: {validation_accuracy}')

            results_df.loc[len(results_df)] = {'id': feedback.id,
                                               'cell_spec': formatted_spec, 
                                               'dna': feedback.dna, 
                                               'val_acc': validation_accuracy, 
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


    def test_nas_algo(self, algo_name, dataset, max_train_hours):
        SEARCH_SPACE = 'sss'    

        nats_bench.api_utils.reset_file_system('default')
        nats_api = nats_bench.create(DEFAULT_NATS_FILEs[SEARCH_SPACE], SEARCH_SPACE, fast_mode=True, verbose=False)

        search_model = self.__get_search_space(SEARCH_SPACE)
        reporting_epoch = DEFAULT_REPORTING_EPOCH[SEARCH_SPACE]

        algorithm = self.__get_algorithm(algo_name)

        results_df = self.__search(nats_api, search_model, algorithm, dataset, reporting_epoch, max_train_hours)

        sorted_results = results_df.sort_values(by='val_acc', ascending=False)

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