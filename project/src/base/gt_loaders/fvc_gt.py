from src.base.gt_loaders.pyb_fvc_gt import PybFVCGTLoader
from src.base.gt_loaders.gt_names import GTName
from src.m_utils import constants as cts

class FVC_GTLoader(PybFVCGTLoader):
    def __init__(self, aligned, ignore_err):
        ground_truth_path = f'{cts.MTEVAL_SUBMODULES_PATH}/icao-reqs-pybossa/ground_truth/'
        super().__init__(GTName.FVC, ground_truth_path, aligned, ignore_err)
        
      


    
    
    
    
