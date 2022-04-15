from src.base.gt_loaders.fvc_gt import FVC_GTLoader
from src.base.gt_loaders.fvc_gt import FVC_GTLoader
from src.base.gt_loaders.pybossa_gt import PybossaGTLoader
from src.base.gt_loaders.genki_gt import Genki4k_GTLoader
from src.base.gt_loaders.imfd_gt import IMFD_GTLoader
from src.base.gt_loaders.cmfd_gt import CMFD_GTLoader
from src.base.gt_loaders.im_search_gt import IMSearch_GTLoader
from src.base.gt_loaders.color_feret_gt import ColorFeret_GTLoader

ALL_GT_LOADERS_LIST = [
    FVC_GTLoader(aligned=False, ignore_err=True), 
    FVC_GTLoader(aligned=True, ignore_err=True),
    #PybossaGTLoader(aligned=True, ignore_err=True),
    #PybossaGTLoader(aligned=False, ignore_err=True),
    #Genki4k_GTLoader(aligned=True, ignore_err=True),
    #IMFD_GTLoader(aligned=True, ignore_err=True),
    #CMFD_GTLoader(aligned=True, ignore_err=True),
    #IMSearch_GTLoader(aligned=True, ignore_err=True),
    #ColorFeret_GTLoader(aligned=True, ignore_err=True)
]