from enum import Enum

class NAS_MTLApproach(Enum):
    APPROACH_1 = 'approach_1'   # fvc + mnist + fashion_mnist -> (Random, RL (bugada)) -> qualificação (nas_v1)
    APPROACH_2 = 'approach_2'   # all_datasets (RL fixed) (nas_v2)
    APPROACH_3 = 'approach_3'   # all_datasets (Random, RL, Evolution) (nas_v3)