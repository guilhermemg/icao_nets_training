import pyglove as pg

from enum import Enum

from src.configs.conf_interp import ConfigInterpreter
from src.nas.v3.rl_dna_generator import RL_DNAGenerator


class NASAlgorithm(Enum):
    RANDOM = 'random'
    EVOLUTION = 'evolution'
    RL = 'rl'


class NASAlgorithmFactory(object):
    def __init__(self, name):
        self.name = name

        print(f' -- Using NAS algorithm: {self.name}\n')      


    def get_algorithm(self, config_interp: ConfigInterpreter = None):
        """Creates algorithm."""
        if self.name.value == NASAlgorithm.RANDOM.value:
            return pg.generators.Random()
        elif self.name.value == NASAlgorithm.EVOLUTION.value:
            return pg.evolution.regularized_evolution(mutator=pg.evolution.mutators.Uniform(), population_size=50, tournament_size=10)
        elif self.name.value == NASAlgorithm.RL.value:
            return RL_DNAGenerator(config_interp)
        else:
            return pg.load(self.name)




    