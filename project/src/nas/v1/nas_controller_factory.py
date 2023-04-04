
from src.nas.v1.nas_controller_1 import NASController_1
from src.nas.v1.nas_controller_2 import NASController_2
from src.base.experiment.training.model_creator import NAS_MTLApproach
from deprecated import deprecated


@deprecated("The NAS v1 is deprecated. Use the NAS v2 or v3 instead.")
class NASControllerFactory:
    
    @staticmethod
    def create_controller(config_interp, model_trainer, model_evaluator, neptune_utils):
        if config_interp.approach.value == NAS_MTLApproach.APPROACH_1.value:
            return NASController_1(model_trainer, model_evaluator, config_interp, neptune_utils)
        elif config_interp.approach.value == NAS_MTLApproach.APPROACH_2.value:
            return NASController_2(model_trainer, model_evaluator, config_interp, neptune_utils)