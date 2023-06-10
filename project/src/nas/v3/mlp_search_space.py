import pyglove as pg

from enum import Enum


class MLPSearchSpace(Enum):
    SS_1 = 'ss_1'



@pg.functor([('n_denses', pg.typing.List(pg.typing.Int())), ('n_convs', pg.typing.List(pg.typing.Int()))])
def model_spec(n_denses, n_convs):
    n_denses_dict = {f'n_denses_{idx}':x for idx,x in enumerate(n_denses)}
    n_convs_dict = {f'n_convs_{idx}':x for idx,x in enumerate(n_convs)}
    return {**n_denses_dict, **n_convs_dict}


class New_MLPSearchSpace(object):

    def __init__(self, ss_indicator=MLPSearchSpace.SS_1):
        self.n_groups = 4
        self.n_denses_candidates = [1, 2, 3, 4, 5]
        self.n_convs_candidates = [1, 2, 3]
        self.ss_indicator = ss_indicator

        print(f' -- Using search space: {self.ss_indicator.value}')


    def get_search_space(self):
        if self.ss_indicator.value == MLPSearchSpace.SS_1.value:
            return model_spec(pg.sublist_of(self.n_groups, self.n_denses_candidates, choices_distinct=False),
                              pg.sublist_of(self.n_groups, self.n_convs_candidates, choices_distinct=False))
        

        