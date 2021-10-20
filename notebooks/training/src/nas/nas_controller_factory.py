
from nas.nas_controller_1 import NASController_1
from nas.nas_controller_2 import NASController_2
from base.model_creator import NAS_MTLApproach

class NASControllerFactory:
    @staticmethod
    def create_controller(approach, model_trainer, model_evaluator, nas_params, neptune_run, use_neptune):
        if approach.value == NAS_MTLApproach.APPROACH_1.value:
            return NASController_1(model_trainer, model_evaluator, nas_params, neptune_run, use_neptune)
        elif approach.value == NAS_MTLApproach.APPROACH_2.value:
            return NASController_2(model_trainer, model_evaluator, nas_params, neptune_run, use_neptune)