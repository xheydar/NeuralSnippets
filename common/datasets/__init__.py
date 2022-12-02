from .mnist import mnist_dataset
from .fashion_mnist import fashion_mnist_dataset
from .cifar10 import cifar10_dataset
from .stanfordcards import stanfordcars_dataset
from .celeba import celeba_dataset

datasets = {}
datasets['mnist'] = mnist_dataset
datasets['fashion_mnist'] = fashion_mnist_dataset
datasets['cifar10'] = cifar10_dataset
datasets['stanfordcars'] = stanfordcars_dataset
datasets['celeba'] = celeba_dataset
