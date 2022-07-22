from enum import Enum


class Eval(Enum):
    COMPLIANT = 1
    NON_COMPLIANT = 0
    DUMMY = -1
    DUMMY_CLS = 2   # dummy value used in classification (-1 is no allowed)
    NO_ANSWER = -99