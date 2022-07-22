
from src.base.experiment.tasks.task import MNIST_TASK
from src.base.experiment.tasks.task import FASHION_MNIST_TASK
from src.base.experiment.tasks.task import CIFAR_10_TASK
from src.base.experiment.tasks.task import CELEB_A_TASK

from enum import Enum

class BenchmarkDataset(Enum):
    MNIST =         {'name': 'mnist',         'target_cols': MNIST_TASK.list_reqs_names()}
    FASHION_MNIST = {'name': 'fashion_mnist', 'target_cols': FASHION_MNIST_TASK.list_reqs_names()}
    CIFAR_10 =      {'name': 'cifar_10',      'target_cols': CIFAR_10_TASK.list_reqs_names()}
    CELEB_A =       {'name': 'celeb_a',       'target_cols': CELEB_A_TASK.list_reqs_names()}