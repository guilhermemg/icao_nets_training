from enum import Enum

class NAS_MTLApproach(Enum):
    APPROACH_1 = 'approach_1'   # fvc + mnist + fashion_mnist -> (Random, RL (bugada)) -> qualificação
    APPROACH_2 = 'approach_2'   # all_datasets (RL fixed)
    APPROACH_3 = 'approach_3'   # all_datasets (Random, RL, Evolution)