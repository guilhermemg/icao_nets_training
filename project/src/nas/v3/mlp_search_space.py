import pyglove as pg

from enum import Enum


class MLPSearchSpaceIndicator(Enum):
    SS_1 = {'name': 'ss_1', 'n_classes': 4}  # just n_denses
    SS_2 = {'name': 'ss_2', 'n_classes': 8}  # n_denses and n_convs


class MLPSearchSpaceCandidate(Enum):
    N_DENSES = [1,2,3,4,5]
    N_CONVS = [1,2,3]


@pg.functor([('n_denses', pg.typing.List(pg.typing.Int())), ('n_convs', pg.typing.List(pg.typing.Int()))])
def model_spec(n_denses=[], n_convs=[]):
    n_denses_dict = {f'n_denses_{idx}':x for idx,x in enumerate(n_denses)}
    n_convs_dict = {f'n_convs_{idx}':x for idx,x in enumerate(n_convs)}
    return {**n_denses_dict, **n_convs_dict}


class New_MLPSearchSpace(object):
    def __init__(self, ss_indicator=MLPSearchSpaceIndicator.SS_1):
        self.n_task_groups = 4
        self.n_denses_candidates = MLPSearchSpaceCandidate.N_DENSES.value
        self.n_convs_candidates = MLPSearchSpaceCandidate.N_CONVS.value
        self.ss_indicator = ss_indicator

        print(f' -- Using search space: {self.ss_indicator.value}')


    def get_search_space(self):
        if self.ss_indicator.name == MLPSearchSpaceIndicator.SS_1.name:
            return model_spec(pg.sublist_of(self.n_task_groups, self.n_denses_candidates, choices_distinct=True))
        elif self.ss_indicator.name == MLPSearchSpaceIndicator.SS_2.name:
            return model_spec(pg.sublist_of(self.n_task_groups, self.n_denses_candidates, choices_distinct=False),
                              pg.sublist_of(self.n_task_groups, self.n_convs_candidates, choices_distinct=False))
        

        