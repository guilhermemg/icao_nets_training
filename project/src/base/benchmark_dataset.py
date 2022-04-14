
from src.m_utils.constants import MNIST_TASK

from enum import Enum

class BenchmarkDataset(Enum):
    MNIST = {'name': 'mnist', 'target_cols': MNIST_TASK.list_reqs_names()}
    FASHION_MNIST = {'name': 'fashion_mnist'}
    CIFAR_10 = {'name': 'cifar_10'}