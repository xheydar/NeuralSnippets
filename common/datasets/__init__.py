from .mnist import mnist_dataset
from .fashion_mnist import fashion_mnist
from .cifar10 import cifar10_dataset

datasets = {}
datasets['mnist'] = mnist_dataset
datasets['fashion_mnist'] = fashion_mnist
datasets['cifar10'] = cifar10_dataset
