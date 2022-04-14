

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as prep_input_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as prep_input_inceptionv3
from tensorflow.keras.applications.vgg19 import preprocess_input as prep_input_vgg19
from tensorflow.keras.applications.vgg16 import preprocess_input as prep_input_vgg16
from tensorflow.keras.applications.resnet_v2 import preprocess_input as prep_input_resnet50v2

from enum import Enum

class BaseModel(Enum):
    MOBILENET_V2 = { 'name':'mobilnet_v2',  'target_size' : (224,224), 'prep_function': prep_input_mobilenetv2 }
    INCEPTION_V3 = { 'name':'inception_v3', 'target_size' : (299,299), 'prep_function': prep_input_inceptionv3 }
    VGG19 =        { 'name':'vgg19',        'target_size' : (224,224), 'prep_function': prep_input_vgg19 }
    VGG16 =        { 'name':'vgg16',        'target_size' : (224,224), 'prep_function': prep_input_vgg16 }
    RESNET50_V2 =  { 'name':'resnet_v2',    'target_size' : (224,224), 'prep_function': prep_input_resnet50v2 }
    CUSTOM =       { 'name':'custom',       'target_size' : (224,224), 'prep_function': prep_input_vgg16}
