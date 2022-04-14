import os
from enum import Enum

SEED = 42

BASE_PATH_0 = '/home/guilherme/data1/Dropbox' if os.path.isdir('/home/guilherme/data1') else '/home/guilherme/Dropbox'
BASE_PATH = BASE_PATH_0 + '/Link to Desktop/Doutorado/Datasets'
ICAO_DATASET_PATH = f'{BASE_PATH}/icao_dataset'
ICAO_CLASSES_PATH = f'{ICAO_DATASET_PATH}/icao_classes.csv'
BASE_OUTPUT_PATH = f'{BASE_PATH_0}/Anaconda Envs Backups/mteval-icao-reqs/icao_reqs'

TEACHABLE_DATASET_PATH = f'{BASE_PATH}/icao_dataset_teachable'

DARKNET_PATH = '/home/guilherme/darknet/'
MTEVAL_BASE_PATH = '/home/guilherme/anaconda3/envs/mteval-icao-reqs'
MTEVAL_UTILS_PATH = f'{MTEVAL_BASE_PATH}/notebooks/utils'
MTEVAL_SUBMODULES_PATH = f'{MTEVAL_BASE_PATH}/submodules'

LABELS_FQA_SCORES_DATA_PATH = f'{BASE_PATH_0}/Anaconda Envs Backups/mteval-icao-reqs/fqa_analysis/labels_fqa_scores.csv'

# class GT_CHECK(Enum):
#     BASE_OUTPUT_PATH = f'{BASE_PATH_0}/Anaconda Envs Backups/mteval-icao-reqs/gt_check'
#     COMPLIANT = 'compliant'
#     NON_COMPLIANT = 'non_compliant'
    

class OUTPUT_PATH(Enum):
    ROTATION_FVC_PATH = f'{BASE_OUTPUT_PATH}/rotation/rotation_fvc.csv'
    ROTATION_VGG_ALIGN_PATH = f'{BASE_OUTPUT_PATH}/rotation/rotation_vgg_align.csv'
    
    OPENFACE_GAZE_PATH = f'{BASE_OUTPUT_PATH}/openface_gaze'
    OPENFACE_GAZE_OUTDIR = f'{OPENFACE_GAZE_PATH}/outdir/'
    OPENFACE_GAZE_FVC_PATH = f'{OPENFACE_GAZE_PATH}/gaze_fvc.csv'
    OPENFACE_GAZE_VGG_PATH = f'{OPENFACE_GAZE_PATH}/gaze_vgg.csv'
    
    OPENFACE_ROTATION_PATH = f'{BASE_OUTPUT_PATH}/openface_rotation'
    OPENFACE_ROTATION_OUTDIR = f'{OPENFACE_ROTATION_PATH}/outdir/'
    OPENFACE_ROTATION_FVC_PATH = f'{OPENFACE_ROTATION_PATH}/rotation_fvc.csv'
    OPENFACE_ROTATION_VGG_PATH = f'{OPENFACE_ROTATION_PATH}/rotation_vgg.csv'
    
    OPENFACE_EYES_CLOSED_PATH = f'{BASE_OUTPUT_PATH}/openface_eyes_closed'
    OPENFACE_EYES_CLOSED_OUTDIR = f'{OPENFACE_EYES_CLOSED_PATH}/outdir/'
    OPENFACE_EYES_CLOSED_FVC_PATH = f'{OPENFACE_EYES_CLOSED_PATH}/eyes_closed_fvc.csv'
    OPENFACE_EYES_CLOSED_VGG_PATH = f'{OPENFACE_EYES_CLOSED_PATH}/eyes_closed_vgg.csv'
    
#     OPENFACE_MOUTH_PATH = f'{BASE_OUTPUT_PATH}/openface_mouth'
#     OPENFACE_MOUTH_OUTDIR = f'{OPENFACE_MOUTH_PATH}/outdir/'
#     OPENFACE_MOUTH_FVC_PATH = f'{OPENFACE_MOUTH_PATH}/mouth_fvc.csv'
#     OPENFACE_MOUTH_VGG_PATH = f'{OPENFACE_MOUTH_PATH}/mouth_vgg.csv'
#     OPENFACE_MOUTH_CALTECH_PATH = f'{OPENFACE_MOUTH_PATH}/mouth_caltech.csv'
    
    OPENFACE_FLASH_LENSES_PATH = f'{BASE_OUTPUT_PATH}/openface_flash_lenses'
    OPENFACE_FLASH_LENSES_OUTDIR = f'{OPENFACE_FLASH_LENSES_PATH}/outdir/'
    
    OPENFACE_RED_EYES_PATH = f'{BASE_OUTPUT_PATH}/openface_red_eyes'
    OPENFACE_RED_EYES_OUTDIR = f'{OPENFACE_RED_EYES_PATH}/outdir/'
    
    MOUTH_FVC_PATH = f'{BASE_OUTPUT_PATH}/mouth/mouth_fvc.csv'
    MOUTH_VGG_ALIGN_PATH = f'{BASE_OUTPUT_PATH}/mouth/mouth_vgg_align.csv'
    
    YOLOV3_CLOSE_PATH = f'{BASE_OUTPUT_PATH}/yolov3_close'    
    YOLOV3_CLOSE_FVC_PATH = f'{YOLOV3_CLOSE_PATH}/close_fvc.csv'
    YOLOV3_CLOSE_VGG_PATH = f'{YOLOV3_CLOSE_PATH}/close_vgg.csv'
    
    YOLOV3_CLOSE_PATH_2 = f'{BASE_OUTPUT_PATH}/yolov3_close_2'    
    YOLOV3_CLOSE_FVC_PATH_2 = f'{YOLOV3_CLOSE_PATH_2}/close_fvc.csv'
    YOLOV3_CLOSE_VGG_PATH_2 = f'{YOLOV3_CLOSE_PATH_2}/close_vgg.csv'
    
    DARKNET_CLOSE_PATH = f'{BASE_OUTPUT_PATH}/darknet_close'
    DARKNET_CLOSE_FVC_PATH = f'{DARKNET_CLOSE_PATH}/close_fvc.csv'
    DARKNET_CLOSE_VGG_PATH = f'{DARKNET_CLOSE_PATH}/close_vgg.csv'
    
    DARKNET_HAT_PATH = f'{BASE_OUTPUT_PATH}/darknet_hat'
    DARKNET_HAT_FVC_PATH = f'{DARKNET_HAT_PATH}/hat_fvc.csv'
    DARKNET_HAT_VGG_PATH = f'{DARKNET_HAT_PATH}/hat_vgg.csv'
    
    DARKNET_DARK_GLASSES_PATH = f'{BASE_OUTPUT_PATH}/darknet_dark_glasses'
    DARKNET_DARK_GLASSES_FVC_PATH = f'{DARKNET_DARK_GLASSES_PATH}/dark_glasses_fvc.csv'
    DARKNET_DARK_GLASSES_VGG_PATH = f'{DARKNET_DARK_GLASSES_PATH}/dark_glasses_vgg.csv'
    
    DARKNET_GLASSES_PATH = f'{BASE_OUTPUT_PATH}/darknet_glasses'
    DARKNET_GLASSES_FVC_PATH = f'{DARKNET_GLASSES_PATH}/glasses_fvc.csv'
    DARKNET_GLASSES_VGG_PATH = f'{DARKNET_GLASSES_PATH}/glasses_vgg.csv'
    
    MASKDETECTOR_VEIL_PATH = f'{BASE_OUTPUT_PATH}/maskdetector_veil'
    MASKDETECTOR_VEIL_FVC_PATH = f'{MASKDETECTOR_VEIL_PATH}/veil_fvc.csv'
    MASKDETECTOR_VEIL_VGG_PATH = f'{MASKDETECTOR_VEIL_PATH}/veil_vgg.csv'
    
    FACEQNET_PATH = f'{BASE_OUTPUT_PATH}/faceqnet'
    FACEQNET_FVC_PATH = f'{FACEQNET_PATH}/faceqnet_fvc.csv'
    FACEQNET_VGG_PATH = f'{FACEQNET_PATH}/faceqnet_vgg.csv'
    
    SPEC_REMOVAL_FLASH_LENSES_PATH = f'{BASE_OUTPUT_PATH}/spec_removal_flash_lenses'
    SPEC_REMOVAL_FLASH_LENSES_FVC_PATH = f'{SPEC_REMOVAL_FLASH_LENSES_PATH}/flash_lenses_fvc.csv'
    SPEC_REMOVAL_FLASH_LENSES_VGG_PATH = f'{SPEC_REMOVAL_FLASH_LENSES_PATH}/flash_lenses_vgg.csv'
    
    FFT_BLURDETECTOR_PATH = f'{BASE_OUTPUT_PATH}/fft_blur'
    FFT_BLURDETECTOR_FVC_PATH = f'{FFT_BLURDETECTOR_PATH}/blur_fvc.csv'
    FFT_BLURDETECTOR_VGG_PATH = f'{FFT_BLURDETECTOR_PATH}/blur_vgg.csv'
    
    
# class MODEL(Enum):
#     MOUTH_EMOPY = 'mouth'
#     MOUTH_OPENFACE = 'openface_mouth'
    
#     ROTATION_HOPENETLITE = 'rotation'
#     ROTATION_OPENFACE = 'openface_rotation'
    
#     GAZE_OPENFACE = 'openface_gaze'
    
#     EYES_CLOSED_OPENFACE = 'openface_eyes_closed'
    
#     CLOSE_YOLOV3 = 'yolov3_close'
    
#     HAT_DARKNET = 'darknet_hat'
#     CLOSE_DARKNET = 'darknet_close'
#     DARK_GLASSES_DARKNET = 'darknet_glasses'
#     GLASSES_DARKNET = 'darknet_glasses'
    
#     VEIL_MASKDETECTOR = 'maskdetector_veil'
    
#     FLASH_LENSES_OPENFACE = 'openface_flash_lenses'
    
#     FACEQNET = 'faceqnet'
    
#     FLASH_LENSES_SPEC_REMOVAL = 'spec_removal_flash_lenses'
    
#     BLUR_DETECTOR_FFT = 'fft_blur'
    
#     CLOSE_YOLOV3_2 = 'yolov3_close_2'
    


class TASK(Enum):
    @classmethod
    def list_reqs_names(cls):
        return [v.value for k,v in cls.__members__.items()]
    
    @classmethod
    def list_reqs(cls):
        return [v for k,v in cls.__members__.items()]


class ICAO_REQ(TASK):
    MOUTH = 'mouth'
    ROTATION = 'rotation'
    L_AWAY = 'l_away'
    EYES_CLOSED = 'eyes_closed'
    CLOSE = 'close'
    HAT = 'hat'
    DARK_GLASSES = 'dark_glasses'
    FRAMES_HEAVY = 'frames_heavy'
    FRAME_EYES = 'frame_eyes'
    FLASH_LENSES = 'flash_lenses'
    VEIL = 'veil'
    REFLECTION = 'reflection'
    LIGHT = 'light'
    SHADOW_FACE = 'sh_face'
    SHADOW_HEAD = 'sh_head'
    BLURRED = 'blurred'
    INK_MARK = 'ink_mark'
    SKIN_TONE = 'skin_tone'
    WASHED_OUT = 'washed_out'
    PIXELATION = 'pixelation'
    HAIR_EYES = 'hair_eyes'
    BACKGROUND = 'background'
    RED_EYES = 'red_eyes'


class MNIST_TASK(TASK):
    N_0 = 'n_0'
    N_1 = 'n_1'
    N_2 = 'n_2'
    N_3 = 'n_3'
    N_4 = 'n_4'
    N_5 = 'n_5'
    N_6 = 'n_6'
    N_7 = 'n_7'
    N_8 = 'n_8'
    N_9 = 'n_9'
    