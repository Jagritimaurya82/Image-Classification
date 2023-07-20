# Image-Classification
This repository contains tutorials covering image classification using [PyTorch](https://github.com/pytorch/pytorch)
 1.7, [torchvision](https://github.com/pytorch/vision) 0.8, [matplotlib](https://matplotlib.org/) 3.3 and [scikit-learn](https://scikit-learn.org/stable/index.html) 0.24, with Python 3.8.

Firstly, we will implement a multilayer perceptron (MLP), then proceed with architectures that utilize convolutional neural networks (CNNs) such as [LeNet](http://yann.lecun.com/exdb/lenet/), [AlexNet](https://www.mathworks.com/help/deeplearning/ref/alexnet.html), [VGG](https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c), and [ResNet](https://arxiv.org/abs/1512.03385).

#### Don't hesitate to [submit an issue](https://github.com/Jagritimaurya82/Image-Classification/issues) if you find any mistakes or disagree with any of the explanations. I'm open to feedback, whether it's positive or negative!

# Getting Started
To install PyTorch, see installation instructions on the [PyTorch](https://pytorch.org/) website.
The instructions to install PyTorch should also detail how to install torchvision but can also be installed using Python Package Index (PyPI):<br>
Install torchvision :[`pip install torchvision`](https://pypi.org/project/torchvision/)<br>
Install matplotlib :[`pip install matplotlib`](https://pypi.org/project/matplotlib/)<br>
Install scikit-learn :[`pip install scikit-learn`](https://pypi.org/project/scikit-learn/)<br>
Install numpy :[`pip install numpy`](https://pypi.org/project/numpy/)<br>
 
 


 
## 1-[Multilayer Perceptron](https://github.com/Jagritimaurya82/Image-Classification/blob/main/1_MLP_MNIST.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jagritimaurya82/Image-Classification/blob/main/1_MLP_MNIST.ipynb)

Training a Multilayer Perceptron (MLP) on the MNIST dataset is a common introductory task in the field of deep learning and computer vision. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0 to 9) and is widely used for digit recognition tasks. Here's a step-by-step guide to train an MLP on the MNIST dataset using Python and popular deep learning libraries like TensorFlow and Keras:
 
 

## 2-[LeNet](https://github.com/Jagritimaurya82/Image-Classification/blob/main/1_LeNet_MNIST.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jagritimaurya82/Image-Classification/blob/main/1_LeNet_MNIST.ipynb)

LeNet, also known as LeNet-5, is a pioneering convolutional neural network (CNN) architecture developed by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner in 1998. It was designed for handwritten digit recognition and is considered one of the early successful CNN models. LeNet played a crucial role in popularizing the use of neural networks for computer vision tasks. The architecture is relatively simple compared to modern CNNs. It consist of two convolutional layers, each followed by a subsampling layer, and then three fully connected linear layes.

## 3-[AlexNet](https://github.com/Jagritimaurya82/Image-Classification/blob/main/3_AlexNet_CIFAR10.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jagritimaurya82/Image-Classification/blob/main/3_AlexNet_CIFAR10.ipynb)

AlexNet is a deep convolutional neural network architecture that was introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, marking a significant breakthrough in computer vision and deep learning. The architecture was named after Alex Krizhevsky, one of the main authors.
Architecture: AlexNet is a deep neural network consisting of eight layers. The first five layers are convolutional layers, followed by three fully connected layers


## 3-[VGG](https://github.com/Jagritimaurya82/Image-Classification/blob/main/4_VGG_Kaggle_CUB200.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jagritimaurya82/Image-Classification/blob/main/4_VGG_Kaggle_CUB200.ipynb)

VGG, short for Visual Geometry Group, is a deep convolutional neural network architecture proposed by researchers from the University of Oxford in 2014. The VGG architecture is known for its simplicity and uniformity, and it played a significant role in advancing the field of computer vision by achieving outstanding performance on image recognition tasks.
The main idea behind VGG is to use very small convolutional filters (3x3) with a relatively deep architecture to learn hierarchical representations of visual features from input images.



# References
While creating these tutorials, I referred to some resources, which might not be up-to-date now. Here is a list of those resources.
* [https://github.com/pytorch/tutorials](https://github.com/pytorch/tutorials)
* [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
