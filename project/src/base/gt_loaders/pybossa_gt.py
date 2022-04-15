from src.base.gt_loaders.pyb_fvc_gt import PybFVCGTLoader
from src.base.gt_loaders.gt_names import GTName
from src.m_utils import constants as cts

class PybossaGTLoader(PybFVCGTLoader):
    def __init__(self, aligned, ignore_err):
        ground_truth_path = f'{cts.MTEVAL_SUBMODULES_PATH}/icao-reqs-pybossa/ground_truth/pybossa/ground_truth/1_2_3_5_6_weight_2/'
        super().__init__(GTName.PYBOSSA, ground_truth_path, aligned, ignore_err)

    
