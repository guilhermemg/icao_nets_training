from enum import Enum


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


class FASHION_MNIST_TASK(TASK):
    TSHIRT = 'tshirt'
    TROUSER = 'trouser'
    PULLOVER = 'pullover'
    DRESS = 'dress'
    COAT = 'coat'
    SANDAL = 'sandal'
    SHIRT = 'shirt'
    SNEAKER = 'sneaker'
    BAG = 'bag'
    ANKLE_BOOT = 'ankle_boot'


class CIFAR_10_TASK(TASK):
    AIRPLANE = 'airplane'
    AUTOMOBILE = 'automobile'
    BIRD = 'bird'
    CAT = 'cat'
    DEER = 'deer'
    DOG = 'dog'
    FROG = 'frog'
    HORSE = 'horse'
    SHIP = 'ship'
    TRUCK = 'truck'


class CELEB_A_TASK(TASK):
    FIVE_OCLOCK_SHADOW = '5_o_Clock_Shadow'
    ARCHED_EYEBROWS = 'Arched_Eyebrows'
    ATTRACTIVE = 'Attractive'
    BAGS_UNDER_EYES = 'Bags_Under_Eyes'
    BALD = 'Bald'
    BANGS = 'Bangs'
    BIG_LIPS = 'Big_Lips'
    BIG_NOSE = 'Big_Nose'
    BLACK_HAIR = 'Black_Hair'
    BLOND_HAIR = 'Blond_Hair'
    BLURRY = 'Blurry'
    BROWN_HAIR = 'Brown_Hair'
    BUSHY_EYEBROWS = 'Bushy_Eyebrows'
    CHUBBY = 'Chubby'
    DOUBLE_CHIN = 'Double_Chin'
    EYEGLASSES = 'Eyeglasses'
    GOATEE = 'Goatee'
    GRAY_HAIR = 'Gray_Hair'
    HEAVY_MAKEUP = 'Heavy_Makeup'
    HIGH_CHEEKBONES = 'High_Cheekbones'
    MALE = 'Male'
    MOUTH_SLIGHTLY_OPEN = 'Mouth_Slightly_Open'
    MUSTACHE = 'Mustache'
    NARROW_EYES = 'Narrow_Eyes'
    NO_BEARD = 'No_Beard'
    OVAL_FACE = 'Oval_Face'
    PALE_SKIN = 'Pale_Skin'
    POINTY_NOSE = 'Pointy_Nose'
    RECEDING_HAIRLINE = 'Receding_Hairline'
    ROSY_CHEEKS = 'Rosy_Cheeks'
    SIDEBURNS = 'Sideburns'
    SMILING = 'Smiling'
    STRAIGHT_HAIR = 'Straight_Hair'
    WAVY_HAIR = 'Wavy_Hair'
    WEARING_EARRINGS = 'Wearing_Earrings'
    WEARING_HAT = 'Wearing_Hat'
    WEARING_LIPSTICK = 'Wearing_Lipstick'
    WEARING_NECKLACE = 'Wearing_Necklace'
    WEARING_NECKTIE = 'Wearing_Necktie'
    YOUNG = 'Young'

    