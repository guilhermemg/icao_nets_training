import pandas as pd
import seaborn as sns

from enum import Enum

from src.m_utils import constants as cts
from src.base.gt_loaders.gen_gt import Eval


class VSOFT_ICAO_PATH(Enum):
    VSOFT_ICAO_BASE_PATH = f'{cts.BASE_OUTPUT_PATH}/vsoft_icao/'
    TRAIN_PATH = f'{VSOFT_ICAO_BASE_PATH}/train.csv'
    VALIDATION_PATH = f'{VSOFT_ICAO_BASE_PATH}/validation.csv'

class VsoftGT:
    def __init__(self, aligned):
        self.name = 'vsoft'
        self.aligned = aligned
        self.ground_truth_df = None
        self.__load_data()
    
    def __load_data(self):
        train = pd.read_csv(VSOFT_ICAO_PATH.TRAIN_PATH.value)
        val = pd.read_csv(VSOFT_ICAO_PATH.VALIDATION_PATH.value)

        print('train.shape: ', train.shape)
        print('val.shape: ', val.shape)
    
        print(train.columns)
    
        cols = {'Imagem':'img_name', 
                'Blurred': cts.ICAO_REQ.BLURRED.value,
                'LookingAway': cts.ICAO_REQ.L_AWAY.value,
                'InkMarked' : cts.ICAO_REQ.INK_MARK.value,
                'UnnaturalSkin': cts.ICAO_REQ.SKIN_TONE.value,
                'TooDarkLight': cts.ICAO_REQ.LIGHT.value,
                'WashedOut': cts.ICAO_REQ.WASHED_OUT.value,
                'Pixelation': cts.ICAO_REQ.PIXELATION.value,
                'HairAcrossEyes': cts.ICAO_REQ.HAIR_EYES.value,
                'EyesClosed': cts.ICAO_REQ.EYES_CLOSED.value,
                'VariedBackground': cts.ICAO_REQ.BACKGROUND.value,
                'Roll_Pitch_Yaw': cts.ICAO_REQ.ROTATION.value,
                'FlashReflectionOnSkin': cts.ICAO_REQ.REFLECTION.value,
                'RedEyes': cts.ICAO_REQ.RED_EYES.value,
                'ShadowsBehindHead': 'sh_head',
                'ShadowsAcrossFace': 'sh_face',
                'DarkTintedLenses': cts.ICAO_REQ.DARK_GLASSES.value,
                'FlashReflectionOnLenses': cts.ICAO_REQ.FLASH_LENSES.value,
                'FramesTooHeavy': cts.ICAO_REQ.FRAMES_HEAVY.value,
                'FramesCoveringEyes': cts.ICAO_REQ.FRAME_EYES.value,
                'Hat_Cap': cts.ICAO_REQ.HAT.value,
                'VeilOverFace': cts.ICAO_REQ.VEIL.value,
                'MouthOpen': cts.ICAO_REQ.MOUTH.value,
                'OtherFaces': cts.ICAO_REQ.CLOSE.value
                }
        
        train.rename(columns=cols, inplace=True)
        val.rename(columns=cols, inplace=True)
                
        train['img_name'] = f"{cts.ICAO_DATASET_PATH}/" + train.img_name.apply(lambda x : x.split('\\')[2] + '/' + x.split('\\')[3])
        val['img_name'] = f"{cts.ICAO_DATASET_PATH}/" + val.img_name.apply(lambda x : x.split('\\')[2] + '/' + x.split('\\')[3])
        
        for col in train.columns:
            if col != 'img_name':
                train[col] = train[col].apply(lambda x : Eval.NON_COMPLIANT.value if x == False else Eval.COMPLIANT.value)
                val[col] = val[col].apply(lambda x : Eval.NON_COMPLIANT.value if x == False else Eval.COMPLIANT.value)
    
        print(train.shape)
        print(val.shape)
    
        self.ground_truth_df = pd.concat([train,val], axis=0)

    def count_images_per_class(self):
        labels_df = self.get_icao_labels()
        cols = ['Compliant','Dummy','NonCompliant','Total']
        final_df = pd.DataFrame(columns=cols, index=labels_df.label_name.values)
        for label in labels_df.label_name:
            comp = self.ground_truth_df[self.ground_truth_df[label] == Eval.COMPLIANT.value][label].count()
            dummy = self.ground_truth_df[self.ground_truth_df[label] == Eval.DUMMY.value][label].count()
            non_comp = self.ground_truth_df[self.ground_truth_df[label] == Eval.NON_COMPLIANT.value][label].count()
            
            final_df.at[label, 'Compliant'] = comp
            final_df.at[label, 'Dummy'] = dummy
            final_df.at[label, 'NonCompliant'] = non_comp
            final_df.at[label, 'Total'] = comp + dummy + non_comp
            
        final_df = final_df[cols].astype(int)    
        
        cm = sns.light_palette("green", as_cmap=True)
        return final_df.style.background_gradient(cmap=cm)

    def get_icao_labels(self):
        return pd.read_csv(cts.ICAO_CLASSES_PATH)
    
