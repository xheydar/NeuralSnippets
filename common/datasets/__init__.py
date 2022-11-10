from .mnist import mnist_dataset
from .cifar10 import cifar10_dataset

datasets = {}
datasets['mnist'] = mnist_dataset
datasets['cifar10'] = cifar10_dataset
