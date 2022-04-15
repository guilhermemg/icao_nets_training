from src.base.data_loaders.fvc_pyb_loader import FvcPybossaDL
from src.base.data_loaders.vgg_loader import VggFace2DL
from src.base.data_loaders.caltech_loader import CaltechDL
from src.base.data_loaders.cvl_loader import CvlDL
from src.base.data_loaders.colorferet_loader import ColorFeretDL
from src.base.data_loaders.fei_loader import FeiDB_DL
from src.base.data_loaders.gtech_loader import GeorgiaTechDL
from src.base.data_loaders.uni_essex_loader import UniEssexDL
from src.base.data_loaders.icpr04_loader import ICPR04_DL
from src.base.data_loaders.imfdb_loader import IMFDB_DL
from src.base.data_loaders.ijbc_loader import IJBC_DL
from src.base.data_loaders.lfw_loader import LFWDL
from src.base.data_loaders.celeba_loader import CelebA_DL
from src.base.data_loaders.casia_webface_loader import CasiaWebface_DL
from src.base.data_loaders.genki_loader import Genki4kDB_DL

ALL_DATA_LOADERS_LIST = [
                           FvcPybossaDL(aligned=False), FvcPybossaDL(aligned=True),
                           #CaltechDL(aligned=False), CaltechDL(aligned=True), 
                           #VggFace2DL(aligned=False), VggFace2DL(aligned=True),
                           #CvlDL(aligned=False), CvlDL(aligned=True), 
                           #ColorFeretDL(aligned=False), ColorFeretDL(aligned=True), 
                           #FeiDB_DL(aligned=False), FeiDB_DL(aligned=True),
                           #GeorgiaTechDL(aligned=False), GeorgiaTechDL(aligned=True), 
                           #UniEssexDL(aligned=False), UniEssexDL(aligned=True),
                           #ICPR04_DL(aligned=False), ICPR04_DL(aligned=True), 
                           #IMFDB_DL(aligned=True),
                           #IJBC_DL(aligned=False), IJBC_DL(aligned=True), 
                           #LFWDL(aligned=False), LFWDL(aligned=True), 
                           CelebA_DL(aligned=True),
                           #CasiaWebface_DL(aligned=False), CasiaWebface_DL(aligned=True),
                           Genki4kDB_DL(aligned=False), Genki4kDB_DL(aligned=True)
                         ]