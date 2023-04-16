import pyglove as pg


class NASAlgorithm(object):
    def __init__(self, name) -> None:
        self.name : str = name
    

    def get_algorithm(self):
        """Creates algorithm."""
        if self.name == 'random':
            return pg.generators.Random()
        elif self.name == 'evolution':
            return pg.evolution.regularized_evolution(mutator=pg.evolution.mutators.Uniform(), population_size=50, tournament_size=10)
        else:
            return pg.load(self.name)