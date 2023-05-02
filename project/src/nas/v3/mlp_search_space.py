import pyglove as pg

from enum import Enum


class MLPSearchSpace(Enum):
    SS_1 = 'ss_1'



@pg.functor([('n_denses', pg.typing.List(pg.typing.Int()))])
def model_1_spec(n_denses):
    return {f'n_denses_{idx}':x for idx,x in enumerate(n_denses)}


class New_MLPSearchSpace(object):

    def __init__(self, ss_indicator=MLPSearchSpace.SS_1):
        self.n_groups = 4
        self.n_denses_candidates = [1, 2, 3, 4, 5]
        self.ss_indicator = ss_indicator

        print(f' -- Using search space: {self.ss_indicator.value}')


    def get_search_space(self):
        """The default search space in NATS-Bench.
    
        Args:
        ss_indicator: tss or sss, indicating the topology or size search space.
    
        Returns:
        A hyper model object that repesents a search space.
        """
        # #info = nats_bench.search_space_info('nats-bench', ss_indicator)
        # #print(info)  # 'candidates': [8, 16, 24, 32, 40, 48, 56, 64], 'num_layers': 5
        # if ss_indicator == 'tss':
        #     total = info['num_nodes'] * (info['num_nodes'] - 1) // 2
        #     return model_tss_spc(pg.sublist_of(total, info['op_names'], choices_distinct=False), info['num_nodes'])
        # elif ss_indicator == 'sss':
        #     #return model_sss_spc(pg.sublist_of(info['num_layers'], info['candidates'], choices_distinct=False))
        #     return model_sss_spc(pg.sublist_of(5, [8, 16, 24, 32, 40, 48, 56, 64], choices_distinct=False))
        if self.ss_indicator.value == MLPSearchSpace.SS_1.value:
            return model_1_spec(pg.sublist_of(self.n_groups, self.n_denses_candidates, choices_distinct=False))

        