from enum import Enum

class Optimizer(Enum):
    ADAM = 'Adam'
    ADAM_CUST = 'AdamCustomized'
    SGD = 'SGD'
    SGD_CUST = 'SGDCustomized'
    SGD_NESTEROV = 'SGDNesterov'
    ADAMAX = 'Adamax'
    ADAMAX_CUST = 'AdamaxCustomized'
    ADAGRAD = 'Adagrad'
    ADAGRAD_CUST = 'AdagradCustomized'
    ADADELTA = 'Adadelta'