
from src.base.experiment.tasks.task import MNIST_TASK
from src.base.experiment.tasks.task import FASHION_MNIST_TASK
from src.base.experiment.tasks.task import CIFAR_10_TASK
from src.base.experiment.tasks.task import CELEB_A_TASK
from src.base.experiment.tasks.task import ICAO_REQ

from enum import Enum

class Dataset(Enum):
    MNIST =         {'name': 'mnist',         'target_cols': MNIST_TASK.list_reqs_names(),         'tasks': list(MNIST_TASK)}
    FASHION_MNIST = {'name': 'fashion_mnist', 'target_cols': FASHION_MNIST_TASK.list_reqs_names(), 'tasks': list(FASHION_MNIST_TASK)}
    CIFAR_10 =      {'name': 'cifar_10',      'target_cols': CIFAR_10_TASK.list_reqs_names(),      'tasks': list(CIFAR_10_TASK)}
    CELEB_A =       {'name': 'celeb_a',       'target_cols': CELEB_A_TASK.list_reqs_names(),       'tasks': list(CELEB_A_TASK)}
    FVC_ICAO =      {'name': 'fvc_icao',      'target_cols': ICAO_REQ.list_reqs_names(),           'tasks': list(ICAO_REQ)}